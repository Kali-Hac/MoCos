import numpy as np
import tensorflow as tf
import torch
import os, sys
from utils import process_SG as process
from tensorflow.python.layers.core import Dense
from sklearn.preprocessing import label_binarize
from sklearn.cluster import DBSCAN
import collections
from sklearn.metrics import average_precision_score
from sklearn import metrics as mr
import gc
import copy


dataset = ''
probe = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20
ft_size = 3  # originial node feature dimension (D)
time_step = 6  # sequence length (f)

# training params
batch_size = 256
nb_epochs = 100000
patience = 100  # patience for early stopping


tf.app.flags.DEFINE_string('dataset', 'KS20', "Dataset: IAS, KS20, BIWI, CASIA-B or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")  # sequence length (f)
tf.app.flags.DEFINE_string('lr', '0.00035', "learning rate")
tf.app.flags.DEFINE_string('probe', 'probe',
						   "for testing probe")  # "probe" (for KGBD/KS20), "A", "B" (for IAS), "Walking", "Still" (for BIWI)
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery")  # probe and gallery setting for CASIA-B
tf.app.flags.DEFINE_string('patience', '150', "epochs for early stopping")
tf.app.flags.DEFINE_string('mode', 'Train', "Training (Train) or Evaluation (Eval)")
tf.app.flags.DEFINE_string('save_flag', '0',
						   "")  # save model metrics (top-1, top-5. top-10, mAP, CSP loss, mACT, mRCL)
tf.app.flags.DEFINE_string('save_model', '0', "")  # save best model
tf.app.flags.DEFINE_string('batch_size', '256', "")
tf.app.flags.DEFINE_string('model_size', '0', "")  # output model size and computational complexity

tf.app.flags.DEFINE_string('H', '128', "")  # embedding size for node representations
tf.app.flags.DEFINE_string('n_heads', '8', "")  # number of Full-Relation (FR) heads
tf.app.flags.DEFINE_string('L_transformer', '2', "")  # number of MGT layers
tf.app.flags.DEFINE_string('fusion_lambda', '0.5', "")  # fusion coefficient for fusing sub-sequence-level and sub-skeleton-level CSP
tf.app.flags.DEFINE_string('t_1', '0.1', "")  # global temperatures t1
tf.app.flags.DEFINE_string('t_2', '10', "")  # global temperatures t2
tf.app.flags.DEFINE_string('pos_enc', '1', "")  # positional encoding or not
tf.app.flags.DEFINE_string('enc_k', '10', "")  # first K eigenvectors for positional encoding
tf.app.flags.DEFINE_string('rand_flip', '1', "")  # random flipping strategy
tf.app.flags.DEFINE_string('prob_t', '0.0', "")  # probability for masking temporal combinatoiral (skeleton graphs)
tf.app.flags.DEFINE_string('prob_s', '0.5', "")  # probability for masking spatial combinatoiral features (body-joint nodes)
tf.app.flags.DEFINE_string('motif_all', '1', "")

FLAGS = tf.app.flags.FLAGS

# check parameters
if FLAGS.dataset not in ['IAS', 'KGBD', 'KS20', 'BIWI', 'CASIA_B']:
	raise Exception('Dataset must be IAS, KGBD, KS20, BIWI or CASIA B.')
if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'
	if FLAGS.length not in ['40', '50', '60']:
		raise Exception('Length number must be 40, 50 or 60')
else:
	if FLAGS.length not in ['4', '6', '8', '10']:
		raise Exception('Length number must be 4, 6, 8 or 10')
if FLAGS.mode not in ['Train', 'Eval']:
	raise Exception('Mode must be Train or Eval.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset
probe = FLAGS.probe
# optimal paramters in MoCos
if dataset == 'KGBD':
	FLAGS.lr = '0.00035'
	FLAGS.rand_flip = '0'
	FLAGS.prob_s = '0.5'
	FLAGS.prob_t = '0.25'
	FLAGS.fusion_lambda = '0.9'
elif dataset == 'CASIA_B':
	FLAGS.lr = '0.00035'
	FLAGS.rand_flip = '0'
	if FLAGS.patience == '150':
		FLAGS.patience = '100'
	FLAGS.fusion_lambda = '1.0'
else:
	FLAGS.lr = '0.00035'
	if dataset == 'IAS':
		FLAGS.rand_flip = '0'
		if probe == 'A':
			FLAGS.prob_s = '0.5'
			FLAGS.prob_t = '0.1'
			FLAGS.fusion_lambda = '0.75'
		elif probe == 'B':
			FLAGS.prob_s = '0.25'
			FLAGS.prob_t = '0.25'
			FLAGS.fusion_lambda = '0.75'
	elif dataset == 'BIWI':
		if probe == 'Walking':
			FLAGS.prob_s = '0.25'
			FLAGS.prob_t = '0.25'
			FLAGS.fusion_lambda = '0.9'
		elif probe == 'Still':
			FLAGS.prob_s = '0.25'
			FLAGS.prob_t = '0.25'
			FLAGS.fusion_lambda = '0.25'
	elif dataset == 'KS20':
		FLAGS.prob_s = '0.25'
		FLAGS.prob_t = '0.25'
		FLAGS.fusion_lambda = '0.9'

time_step = int(FLAGS.length)
probe = FLAGS.probe
patience = int(FLAGS.patience)
batch_size = int(FLAGS.batch_size)

# not used
global_att = False
nhood = 1
residual = False
nonlinearity = tf.nn.elu

pre_dir = 'ReID_Models/'
# Customize the [directory] to save models with different hyper-parameters
change = ''

if FLAGS.probe_type != '':
	change += '_CME'


try:
	os.mkdir(pre_dir)
except:
	pass

if dataset == 'KS20':
	nb_nodes = 25

if dataset == 'CASIA_B':
	nb_nodes = 14

if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'


print('----- Model hyperparams -----')
print('f (sequence length): ' + str(time_step))
print('H (embedding size): ' + FLAGS.H)
print('MGT Layers: ' + FLAGS.L_transformer)
print('heads: ' + FLAGS.n_heads)
print('MoCos fusion lambda: ' + FLAGS.fusion_lambda)
print('t1: ' + FLAGS.t_1)
print('t2: ' + FLAGS.t_2)

print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))

print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

print('p_s: ' + FLAGS.prob_s)
print('p_t: ' + FLAGS.prob_t)
print('lambda: ' + FLAGS.fusion_lambda)

if dataset == 'CASIA_B':
	# 1-order
	adj_1 = np.zeros([14, 14])
	adj_1.astype(np.int8)
	adj_1[0, 1] = adj_1[1, 0] = 1
	adj_1[1, 2] = adj_1[2, 1] = 1
	adj_1[2, 3] = adj_1[3, 2] = 1
	adj_1[3, 4] = adj_1[4, 3] = 1
	adj_1[1, 5] = adj_1[5, 1] = 1
	adj_1[5, 6] = adj_1[6, 5] = 1
	adj_1[6, 7] = adj_1[7, 6] = 1
	adj_1[1, 8] = adj_1[8, 1] = 1
	adj_1[8, 9] = adj_1[9, 8] = 1
	adj_1[9, 10] = adj_1[10, 9] = 1
	adj_1[1, 11] = adj_1[11, 1] = 1
	adj_1[11, 12] = adj_1[12, 11] = 1
	adj_1[12, 13] = adj_1[13, 12] = 1

	# 2-order
	adj_2 = copy.deepcopy(adj_1)
	adj_2[0, [2, 5, 8, 11]] = 1
	adj_2[1, [3, 6, 9, 12]] = 1
	adj_2[2, [0, 5, 4, 8, 11]] = 1
	adj_2[3, 1] = 1
	adj_2[4, 2] = 1
	adj_2[5, [0, 2, 8, 11, 7]] = 1
	adj_2[6, 1] = 1
	adj_2[7, 5] = 1
	adj_2[8, [2, 0, 5, 11, 10]] = 1
	adj_2[9, 1] = 1
	adj_2[10, 8] = 1
	adj_2[11, [0, 2, 5, 8, 13]] = 1
	adj_2[12, 1] = 1
	adj_2[13, 11] = 1

	# 3-order
	adj_3 = copy.deepcopy(adj_2)
	adj_3[0, [3, 6, 9, 12]] = 1
	adj_3[1, [4, 7, 10, 13]] = 1
	adj_3[2, [6, 9, 12]] = 1
	adj_3[3, [0, 5, 8, 11]] = 1
	adj_3[4, 1] = 1
	adj_3[5, [3, 9, 12]] = 1
	adj_3[6, [0, 2, 8, 11]] = 1
	adj_3[7, [1]] = 1
	adj_3[8, [3, 12, 6]] = 1
	adj_3[9, [0, 2, 11, 5]] = 1
	adj_3[10, [1]] = 1
	adj_3[11, [3, 9, 6]] = 1
	adj_3[12, [0, 2, 5, 8]] = 1
	adj_3[13, [1]] = 1

	# hand + arm
	adj_4 = np.zeros([14, 14])
	adj_4[2, [3, 4]] = 1
	adj_4[3, [2, 4]] = 1
	adj_4[4, [2, 3]] = 1
	for i in [2, 3, 4]:
		for j in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
			adj_4[i, j] = 1
	adj_4[5, [6, 7]] = 1
	adj_4[6, [5, 7]] = 1
	adj_4[7, [5, 6]] = 1
	for i in [5, 6, 7]:
		for j in [2, 3, 4, 8, 9, 10, 11, 12, 13]:
			adj_4[i, j] = 1

	# foot + leg
	adj_5 = np.zeros([14, 14])
	adj_5[8, [9, 10]] = 1
	adj_5[9, [8, 10]] = 1
	adj_5[10, [8, 9]] = 1
	for i in [8, 9, 10]:
		for j in [2, 3, 4, 5, 6, 7, 11, 12, 13]:
			adj_5[i, j] = 1
	adj_5[11, [12, 13]] = 1
	adj_5[12, [11, 13]] = 1
	adj_5[13, [11, 12]] = 1
	for i in [11, 12, 13]:
		for j in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
			adj_5[i, j] = 1

elif dataset != 'KS20':
	# 1-order
	adj_1 = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,],
			  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,]]

	# 2-order
	adj_2 = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
			  [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,],
			  [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,],
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,],
			  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,],
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,]]

	# 3-order
	adj_3 = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,],
			  [1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
			  [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,],
			  [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,],
			  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0,],
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,],
			  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1,],
			  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,],
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,]]

	# hand + arm
	adj_4 = np.zeros([20, 20])
	adj_4[8, [9, 10, 11]] = 1
	adj_4[9, [8, 10, 11]] = 1
	adj_4[10, [9, 8, 11]] = 1
	adj_4[11, [9, 10, 8]] = 1
	for i in [8, 9, 10, 11]:
		for j in [4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15]:
			adj_4[i, j] = 1
	adj_4[4, [5, 6, 7]] = 1
	adj_4[5, [4, 6, 7]] = 1
	adj_4[6, [5, 4, 7]] = 1
	adj_4[7, [5, 6, 4]] = 1
	# #
	for i in [4, 5, 6, 7]:
		for j in [8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15]:
			adj_4[i, j] = 1

	# foot + leg
	adj_5 = np.zeros([20, 20])
	adj_5[16, [17, 18, 19]] = 1
	adj_5[17, [16, 18, 19]] = 1
	adj_5[18, [16, 17, 19]] = 1
	adj_5[19, [16, 18, 17]] = 1
	for i in [16, 17, 18, 19]:
		for j in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
			adj_5[i, j] = 1
	adj_5[12, [13, 14, 15]] = 1
	adj_5[13, [12, 14, 15]] = 1
	adj_5[14, [12, 13, 15]] = 1
	adj_5[15, [12, 13, 14]] = 1
	for i in [12, 13, 14, 15]:
		for j in [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19]:
			adj_5[i, j] = 1
else:
	# 1-order
	adj_1 = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
			 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
			 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,],
			 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
			 [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], # 20
			 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],]

	# 2-order
	adj_2 = copy.deepcopy(adj_1)
	adj_2 = np.array(adj_2)
	adj_2[0, [1, 20, 12, 13, 16, 17]] = 1
	adj_2[1, [20, 2, 4, 8, 0, 12, 16]] = 1
	adj_2[2, [3, 20, 4, 8, 1]] = 1
	adj_2[3, [2, 20]] = 1
	adj_2[4, [5, 6, 20, 2, 8, 1]] = 1
	adj_2[5, [4, 20, 6, 7, 22]] = 1
	adj_2[6, [7, 21, 5, 4]] = 1
	adj_2[7, [21, 6, 22,5]] = 1
	adj_2[8, [20, 2, 4, 1, 9, 10]] = 1
	adj_2[9, [8, 20, 10, 11, 24]] = 1
	adj_2[10, [9, 8, 11, 23]] = 1
	adj_2[11, [10, 9, 24]] = 1
	adj_2[12, [0, 1, 16, 13, 14]] = 1
	adj_2[13, [14, 15, 12, 0]] = 1
	adj_2[14, [13, 12]] = 1
	adj_2[15, [14, 13]] = 1
	adj_2[16, [0, 1, 12, 17, 18]] = 1
	adj_2[17, [16, 0, 18, 19]] = 1
	adj_2[18, [17, 16]] = 1
	adj_2[19, [18, 17]] = 1
	adj_2[20, [4, 5, 2, 3, 8, 9, 1, 0]] = 1
	adj_2[21, [7, 6]] = 1
	adj_2[22, [6, 7, 5]] = 1
	adj_2[23, [11, 10]] = 1
	adj_2[24, [10, 9, 11]] = 1


	# 3-order
	adj_3 = copy.deepcopy(adj_1)
	adj_3 = np.array(adj_3)
	adj_3[0, [1, 20, 2, 4, 8, 12, 13, 14, 16, 17, 18]] = 1
	adj_3[1, [20, 2, 3, 4, 5, 8, 9, 0, 12, 13, 16, 17]] = 1
	adj_3[2, [3, 20, 1, 4, 5, 8, 9, 1, 0]] = 1
	adj_3[3, [2, 20, 4, 8, 1]] = 1
	adj_3[4, [5, 6, 7, 22, 20, 2, 3, 8, 9, 1, 0]] = 1
	adj_3[5, [4, 20, 2, 1, 8, 6, 7, 21, 22]] = 1
	adj_3[6, [7, 21, 5, 4, 20]] = 1
	adj_3[7, [21, 6, 22, 5, 4]] = 1
	adj_3[8, [20, 2, 3, 4, 5, 1, 0, 9, 10, 11, 24]] = 1
	adj_3[9, [8, 20, 2, 4, 1, 10, 11, 23, 24]] = 1
	adj_3[10, [9, 8, 20, 11, 23]] = 1
	adj_3[11, [10, 9, 8, 24]] = 1
	adj_3[12, [0, 1, 20, 16, 17, 13, 14, 15]] = 1
	adj_3[13, [14, 15, 12, 0, 1, 16]] = 1
	adj_3[14, [13, 12, 0]] = 1
	adj_3[15, [14, 13, 12]] = 1
	adj_3[16, [0, 1, 20, 12, 13, 17, 18, 19]] = 1
	adj_3[17, [16, 0, 1, 12, 18, 19]] = 1
	adj_3[18, [17, 16, 0]] = 1
	adj_3[19, [18, 17, 16]] = 1
	adj_3[20, [4, 5, 6, 2, 3, 8, 9, 10, 1, 0, 12, 16]] = 1
	adj_3[21, [7, 6, 5, 22]] = 1
	adj_3[22, [6, 7, 21, 5, 4]] = 1
	adj_3[23, [11, 10, 9, 24]] = 1
	adj_3[24, [10, 9, 8, 11, 23]] = 1

	# hand + arm
	adj_4 = np.zeros([25, 25])
	adj_4[8, [9, 10, 11, 23, 24]] = 1
	adj_4[9, [8, 10, 11, 23, 24]] = 1
	adj_4[10, [8, 9, 11, 23, 24]] = 1
	adj_4[11, [8, 9, 10, 23, 24]] = 1
	adj_4[23, [8, 9, 10, 11, 24]] = 1
	adj_4[24, [8, 9, 10, 11, 23]] = 1
	for i in [8, 9, 10, 11, 23, 24]:
		for j in [4, 5, 6, 7, 21, 22, 16, 17, 18, 19, 12, 13, 14, 15]:
			adj_4[i, j] = 1
	adj_4[4, [5, 6, 7, 21, 22]] = 1
	adj_4[5, [4, 6, 7, 21, 22]] = 1
	adj_4[6, [4, 5, 7, 21, 22]] = 1
	adj_4[7, [4, 5, 6, 21, 22]] = 1
	adj_4[21, [4, 5, 6, 7, 22]] = 1
	adj_4[22, [4, 5, 6, 7, 21]] = 1
	for i in [4, 5, 6, 7, 21, 22]:
		for j in [8, 9, 10, 11, 23, 24, 16, 17, 18, 19, 12, 13, 14, 15]:
			adj_4[i, j] = 1

	# foot + leg
	adj_5 = np.zeros([25, 25])
	adj_5[16, [17, 18, 19]] = 1
	adj_5[17, [16, 18, 19]] = 1
	adj_5[18, [16, 17, 19]] = 1
	adj_5[19, [16, 18, 17]] = 1
	for i in [16, 17, 18, 19]:
		for j in [4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24, 12, 13, 14, 15]:
			adj_5[i, j] = 1
	adj_5[12, [13, 14, 15]] = 1
	adj_5[13, [12, 14, 15]] = 1
	adj_5[14, [12, 13, 15]] = 1
	adj_5[15, [12, 13, 14]] = 1
	for i in [12, 13, 14, 15]:
		for j in [4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24, 16, 17, 18, 19]:
			adj_5[i, j] = 1

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)

if FLAGS.probe_type == '':
	X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, \
	adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
							   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,)
	del _
	gc.collect()

else:
	from utils import process_cme_SG as process

	X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, \
	adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
							   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
							   PG_type=FLAGS.probe_type.split('.')[0])
	print('## [Probe].[Gallery]', FLAGS.probe_type)
	del _
	gc.collect()

all_ftr_size = int(FLAGS.H)
loaded_graph = tf.Graph()
joint_num = X_train_J.shape[2]
cluster_epochs = 15000
display = 20
k = int(FLAGS.enc_k)

change += '_MoCos_Formal_f_' + FLAGS.length  + '_prob_s_' + FLAGS.prob_s + '_prob_t_' + FLAGS.prob_t  + '_lambda_' + FLAGS.fusion_lambda

if FLAGS.mode == 'Train':
	loaded_graph = tf.Graph()
	with loaded_graph.as_default():
		with tf.name_scope('Input'):
			J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, joint_num, ft_size))
			L_eig = tf.placeholder(dtype=tf.float32, shape=(joint_num, k))
			train_flag = tf.placeholder(dtype=tf.bool, shape=())
			pseudo_lab_1 = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_1 = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))
			pseudo_lab_2 = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_2 = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))
			seq_mask = tf.placeholder(dtype=tf.float32, shape=(time_step,))
			seq_mask_2 = tf.placeholder(dtype=tf.float32, shape=(time_step,))
			node_mask = tf.placeholder(dtype=tf.float32, shape=(joint_num,))

			gt_class_ftr = tf.placeholder(dtype=tf.float32, shape=(nb_classes, all_ftr_size))
			gt_lab = tf.placeholder(dtype=tf.int32, shape=(batch_size,))

		with tf.name_scope("MoCos"), tf.variable_scope("MoCos", reuse=tf.AUTO_REUSE):
			inputs = tf.reshape(J_in, [time_step * batch_size * joint_num, 3])
			outputs = inputs
			outputs = tf.layers.dense(outputs, int(FLAGS.H), activation=tf.nn.relu)
			s_rep = outputs
			s_rep = tf.layers.dense(s_rep, int(FLAGS.H), activation=None)
			s_rep = tf.reshape(s_rep, [-1])
			optimizer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.lr))
			seq_ftr = tf.reshape(s_rep, [batch_size, time_step, joint_num, -1])
			pos_enc = tf.tile(tf.reshape(L_eig, [1, 1, joint_num, k]), [batch_size, time_step, 1, 1])
			pos_enc = tf.layers.dense(pos_enc, int(FLAGS.H), activation=None)
			ori_ftr = seq_ftr
			if FLAGS.pos_enc == '1':
				seq_ftr = seq_ftr + pos_enc
			H = int(FLAGS.H)
			W_head = lambda: tf.Variable(tf.random_normal([H, H // int(FLAGS.n_heads)]))

			h = seq_ftr
			for l in range(int(FLAGS.L_transformer)):
				for i in range(int(FLAGS.n_heads)):
					W_Q = tf.Variable(initial_value=W_head)
					W_K = tf.Variable(initial_value=W_head)
					W_V = tf.Variable(initial_value=W_head)
					Q_h = tf.matmul(h, W_Q)
					K_h = tf.matmul(h, W_K)
					K_h = tf.transpose(K_h, perm=[0, 1, 3, 2])
					motif_indice = []
					if i == 1:
						att_scores = tf.nn.softmax(
							tf.clip_by_value(tf.matmul(Q_h, K_h) * adj_1 / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					if i == 2:
						att_scores = tf.nn.softmax(
							tf.clip_by_value(tf.matmul(Q_h, K_h) * adj_2 / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					if i == 3:
						att_scores = tf.nn.softmax(
							tf.clip_by_value(tf.matmul(Q_h, K_h) * adj_3 / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					if i == 4:
						att_scores = tf.nn.softmax(
							tf.clip_by_value(tf.matmul(Q_h, K_h) * adj_4 / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					if i == 5:
						att_scores = tf.nn.softmax(
							tf.clip_by_value(tf.matmul(Q_h, K_h) * adj_5 / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					if i not in [1, 2, 3, 4, 5]:
						# numerical stability to clamp [-5, 5]
						att_scores = tf.nn.softmax(
							tf.clip_by_value(tf.matmul(Q_h, K_h) / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					V_h = tf.matmul(h, W_V)
					att_scores = tf.tile(tf.reshape(att_scores, [batch_size, time_step, joint_num, joint_num, 1]),
										 [1, 1, 1, 1, H // int(FLAGS.n_heads)])
					aggr_features = tf.reduce_sum(
						att_scores * tf.reshape(V_h, [batch_size, time_step, 1, joint_num, H // int(FLAGS.n_heads)]),
						axis=-2)
					if i == 0:
						concat_features = aggr_features
					else:
						concat_features = tf.concat([concat_features, aggr_features], axis=-1)
				h = concat_features

			h = tf.layers.dropout(h, rate=0.5, training=train_flag)
			h = tf.layers.dense(h, H, activation=None)
			h_res1 = seq_ftr
			h = h_res1 + h
			h = tf.layers.batch_normalization(h, training=train_flag)
			h_res2 = h
			h = tf.layers.dense(h, H * 2, activation=tf.nn.relu)
			h = tf.layers.dropout(h, rate=0.5, training=train_flag)
			h = tf.layers.dense(h, H, activation=None)
			h = h_res2 + h
			h = tf.layers.batch_normalization(h, training=train_flag)
			spatial_h = h

			# sub-skeleton-level CSP in MoCos
			def CSP_ske(t, labels, all_ftr, cluster_ftr):
				W_head = lambda: tf.Variable(tf.random_normal([H, H]))
				head_num = 1
				for i in range(head_num):
					f_1 = tf.Variable(initial_value=W_head)
					f_2 = tf.Variable(initial_value=W_head)
					all_ftr_trans = tf.matmul(all_ftr, f_1)
					cluster_ftr_trans = tf.matmul(cluster_ftr, f_2)
					logits = tf.matmul(all_ftr_trans, tf.transpose(cluster_ftr_trans)) / t
					label_frames = tf.reshape(tf.tile(tf.reshape(labels, [-1, 1]), [1, time_step]), [-1])
					label_frames = tf.reshape(label_frames, [batch_size, time_step])
					loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_frames,
																					logits=logits), axis=-1))
				return loss

			ori_spatial_h = spatial_h
			spatial_h = tf.reduce_mean(spatial_h, axis=-2)
			h2 = spatial_h
			C_seq = seq_ftr = h2

			seq_ftr = tf.reduce_mean(seq_ftr, axis=1)
			seq_ftr = tf.reshape(seq_ftr, [batch_size, -1])

			mask_seq_1 = tf.boolean_mask(C_seq, tf.reshape(seq_mask, [-1]), axis=1)
			mask_seq_1 = tf.reduce_mean(mask_seq_1, axis=1)


			# sub-sequence-level CSP in MoCos
			def CSP_seq(t, pseudo_lab, all_ftr, cluster_ftr):
				all_ftr = tf.nn.l2_normalize(all_ftr, axis=-1)
				cluster_ftr = tf.nn.l2_normalize(cluster_ftr, axis=-1)
				output = tf.matmul(all_ftr, tf.transpose(cluster_ftr))
				output /= t
				loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab, logits=output))
				return loss

			mask_G = tf.boolean_mask(ori_spatial_h, tf.reshape(node_mask, [-1]), axis=-2)
			G_h = tf.reduce_mean(mask_G, axis=-2)
			# temporal masking in CSP
			G_mask_seq = tf.boolean_mask(G_h, tf.reshape(seq_mask, [-1]), axis=1)
			G_h_seq = tf.reduce_mean(G_mask_seq, axis=-2)

			SSq_CSP_loss = CSP_seq(float(FLAGS.t_1), gt_lab, G_h_seq, gt_class_ftr)
			SSk_CSP_loss = CSP_ske(float(FLAGS.t_2), gt_lab, G_h, gt_class_ftr)
			CSP_loss = (1 - float(FLAGS.fusion_lambda)) * SSk_CSP_loss + float(FLAGS.fusion_lambda) * SSq_CSP_loss
			train_op = optimizer.minimize(CSP_loss)

		saver = tf.train.Saver()
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session(config=config) as sess:
			sess.run(init_op)
			if FLAGS.model_size == '1':
				# compute model size (M) and computational complexity (GFLOPs)
				def stats_graph(graph):
					flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
					params = tf.profiler.profile(graph,
												 options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
					print('FLOPs: {} GFLOPS;    Trainable params: {} M'.format(flops.total_float_ops / 1e9,
																			   params.total_parameters / 1e6))
				stats_graph(loaded_graph)
				print('prob_s:', FLAGS.prob_s)
				print('prob_t:', FLAGS.prob_t)
				print('fusion_lambda:', FLAGS.fusion_lambda)
				exit()

			mask_rand_save = []
			mask_rand_save_2 = []
			node_mask_save = []
			cur_epoch = 0

			def train_loader(X_train_J, y_train):
				global mask_rand_save, node_mask_save, mask_rand_save_2

				# temporal masking
				mask_rand_save = []
				mask_rand_save_2 = []
				# spatial masking
				node_mask_save = []
				tr_step = 0
				tr_size = X_train_J.shape[0]
				train_labels_all = []
				train_features_all = []
				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					prob = np.random.uniform(0, 1, [time_step, ])
					mask_rand_1 = prob >= float(FLAGS.prob_t)
					# ensure not all zeros
					while np.mean(mask_rand_1) == 0:
						prob = np.random.uniform(0, 1, [time_step, ])
						mask_rand_1 = prob >= float(FLAGS.prob_t)
					mask_rand_1 = np.reshape(mask_rand_1, [time_step])
					mask_rand_save.append(mask_rand_1.tolist())
					prob = np.random.uniform(0, 1, [time_step, ])
					mask_rand_2 = prob >= float(FLAGS.prob_t)
					while np.mean(mask_rand_2) == 0:
						prob = np.random.uniform(0, 1, [time_step, ])
						mask_rand_2 = prob >= float(FLAGS.prob_t)
					mask_rand_2 = np.reshape(mask_rand_2, [time_step])
					mask_rand_save_2.append(mask_rand_2.tolist())
					prob = np.random.uniform(0, 1, [joint_num, ])
					node_mask_rand = prob >= float(FLAGS.prob_s)
					# ensure not all zeros
					while np.mean(node_mask_rand) == 0:
						prob = np.random.uniform(0, 1, [joint_num, ])
						node_mask_rand = prob >= float(FLAGS.prob_s)
					node_mask_save.append(node_mask_rand.tolist())

					[all_features] = sess.run([seq_ftr],
															  feed_dict={
																  J_in: X_input_J,
																  seq_mask: mask_rand_1,
																  L_eig: pos_enc_ori,
																  train_flag: False,
																  node_mask: node_mask_rand,
															  })
					train_features_all.extend(all_features.tolist())
					train_labels_all.extend(labels.tolist())
					tr_step += 1

				train_features_all = np.array(train_features_all).astype(np.float32)
				train_features_all = torch.from_numpy(train_features_all)
				return train_features_all, train_labels_all

			def gal_loader(X_train_J, y_train):
				tr_step = 0
				tr_size = X_train_J.shape[0]
				gal_logits_all = []
				gal_labels_all = []
				gal_features_all = []
				embed_1_all = []
				embed_2_all = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					[Seq_features] = sess.run([seq_ftr],
											  feed_dict={
												  J_in: X_input_J,
												  L_eig: pos_enc_ori,
												  train_flag: False
											  })
					gal_features_all.extend(Seq_features.tolist())
					gal_labels_all.extend(labels.tolist())
					tr_step += 1
				return gal_features_all, gal_labels_all, embed_1_all, embed_2_all

			def evaluation():
				vl_step = 0
				vl_size = X_test_J.shape[0]
				pro_labels_all = []
				pro_features_all = []
				while vl_step * batch_size < vl_size:
					if (vl_step + 1) * batch_size > vl_size:
						break
					X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
					[Seq_features] = sess.run([seq_ftr],
											  feed_dict={
												  J_in: X_input_J,
												  L_eig: pos_enc_ori,
												  train_flag: False
											  })
					pro_labels_all.extend(labels.tolist())
					pro_features_all.extend(Seq_features.tolist())
					vl_step += 1
				X = np.array(gal_features_all)
				y = np.array(gal_labels_all)
				t_X = np.array(pro_features_all)
				t_y = np.array(pro_labels_all)
				t_y = np.argmax(t_y, axis=-1)
				y = np.argmax(y, axis=-1)

				def mean_ap(distmat, query_ids=None, gallery_ids=None,
							query_cams=None, gallery_cams=None):
					# distmat = to_numpy(distmat)
					m, n = distmat.shape
					# Fill up default values
					if query_ids is None:
						query_ids = np.arange(m)
					if gallery_ids is None:
						gallery_ids = np.arange(n)
					if query_cams is None:
						query_cams = np.zeros(m).astype(np.int32)
					if gallery_cams is None:
						gallery_cams = np.ones(n).astype(np.int32)
					# Ensure numpy array
					query_ids = np.asarray(query_ids)
					gallery_ids = np.asarray(gallery_ids)
					query_cams = np.asarray(query_cams)
					gallery_cams = np.asarray(gallery_cams)
					# Sort and find correct matches
					indices = np.argsort(distmat, axis=1)
					matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
					# Compute AP for each query
					aps = []
					if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(1, m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
									 (gallery_cams[indices[i]] != query_cams[i]))

							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							# discard nan
							y_score[np.isnan(y_score)] = 0
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					else:
						for i in range(m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
									 (gallery_cams[indices[i]] != query_cams[i]))
							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					if len(aps) == 0:
						raise RuntimeError("No valid query")
					return np.mean(aps)

				def metrics(X, y, t_X, t_y):
					# compute Euclidean distance
					if dataset != 'CASIA_B':
						a, b = torch.from_numpy(t_X), torch.from_numpy(X)
						m, n = a.size(0), b.size(0)
						a = a.view(m, -1)
						b = b.view(n, -1)
						dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
								 torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
						dist_m.addmm_(1, -2, a, b.t())
						# 1e-12
						dist_m = (dist_m.clamp(min=1e-12)).sqrt()
						mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
						_, dist_sort = dist_m.sort(1)
						dist_sort = dist_sort.numpy()
					else:
						X = np.array(X)
						t_X = np.array(t_X)
						dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_m = np.array(dist_m)
						mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
						dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_sort = np.array(dist_sort)

					top_1 = top_5 = top_10 = 0
					probe_num = dist_sort.shape[0]
					if (FLAGS.probe_type == 'nm.nm' or
							FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(probe_num):
							if t_y[i] in y[dist_sort[i, 1:2]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, 1:6]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, 1:11]]:
								top_10 += 1
					else:
						for i in range(probe_num):
							# print(dist_sort[i, :10])
							if t_y[i] in y[dist_sort[i, :1]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, :5]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, :10]]:
								top_10 += 1
					return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

				mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
				del X, y, t_X, t_y, pro_labels_all, pro_features_all
				gc.collect()
				return mAP, top_1, top_5, top_10

			max_acc_1 = 0
			max_acc_2 = 0
			top_5_max = 0
			top_10_max = 0
			cur_patience = 0
			top_1s = []
			top_5s = []
			top_10s = []
			mAPs = []
			SSq_CSP_losses = []
			SSk_CSP_losses = []
			CSP_losses = []

			mACT = []
			mRCL = []

			if dataset == 'KGBD' or dataset == 'KS20':
				# if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

			elif dataset == 'BIWI':
				if probe == 'Walking':
					# if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				else:
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
			elif dataset == 'IAS':
				if probe == 'A':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				else:
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

			elif dataset == 'CASIA_B':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[1])
			del _
			gc.collect()

			all_strs = ''
			simp_strs = ''
			for epoch in range(cluster_epochs):
				cur_epoch = epoch
				train_features_all, train_labels_all = train_loader(X_train_J, y_train)
				gal_features_all, gal_labels_all, gal_embed_1_all, gal_embed_2_all = gal_loader(X_gal_J, y_gal)
				ori_train_labels = copy.deepcopy(train_labels_all)
				if FLAGS.save_flag == '1':
					# Compute mean intra-class tightness (mACT) and mean inter-class tightness (mRCL)
					# see "Skeleton Prototype Contrastive Learning with Multi-level Graph Relation Modeling
					# for Unsupervised Person Re-Identification" (Rao et al.) for details of above metrics
					train_features_all = train_features_all.numpy()
					labels = np.argmax(np.array(train_labels_all), axis=-1)
					label_t = set(labels.tolist())
					y = np.array(labels)
					X = np.array(train_features_all)
					sorted_indices = np.argsort(y, axis=0)
					sort_y = y[sorted_indices]
					sort_X = X[sorted_indices]
					all_class_ftrs = {}
					class_start_indices = {}
					class_end_indices = {}
					pre_label = sort_y[0]
					class_start_indices[pre_label] = 0
					for i, label in enumerate(sort_y):
						if sort_y[i] not in all_class_ftrs.keys():
							all_class_ftrs[sort_y[i]] = [sort_X[i]]
						else:
							all_class_ftrs[sort_y[i]].append(sort_X[i])
						if label != pre_label:
							class_start_indices[label] = class_end_indices[pre_label] = i
							pre_label = label
						if i == len(sort_y) - 1:
							class_end_indices[label] = i
					center_ftrs = []
					for label, class_ftrs in all_class_ftrs.items():
						class_ftrs = np.array(class_ftrs)
						center_ftr = np.mean(class_ftrs, axis=0)
						center_ftrs.append(center_ftr)
					center_ftrs = np.array(center_ftrs)

					a, b = torch.from_numpy(sort_X), torch.from_numpy(center_ftrs)

					a_norm = a / a.norm(dim=1)[:, None]
					b_norm = b / b.norm(dim=1)[:, None]
					dist_m = 1 - torch.mm(a_norm, b_norm.t())
					dist_m = dist_m.numpy()

					prototype_dis_m = np.zeros([nb_classes, nb_classes])
					for i in range(nb_classes):
						prototype_dis_m[i, :] = np.mean(dist_m[class_start_indices[i]:class_end_indices[i], :], axis=0)

					intra_class_dis = np.mean(prototype_dis_m.diagonal())
					sum_distance = np.reshape(np.sum(prototype_dis_m, axis=-1), [nb_classes, ])
					average_distance = np.sum(sum_distance) / (nb_classes * nb_classes)

					a = b = torch.from_numpy(center_ftrs)
					a_norm = a / a.norm(dim=1)[:, None]
					b_norm = b / b.norm(dim=1)[:, None]
					dist_m = 1 - torch.mm(a_norm, b_norm.t())
					dist_m = dist_m.numpy()
					inter_class_dis = np.mean(dist_m)

					mACT.append(average_distance / intra_class_dis)
					mRCL.append(inter_class_dis / average_distance)
					print('mACT: ', average_distance / intra_class_dis, 'mRCL: ', inter_class_dis / average_distance)
					train_features_all = torch.from_numpy(train_features_all)
				mAP, top_1, top_5, top_10 = evaluation()
				top_1s.append(top_1)
				top_5s.append(top_5)
				top_10s.append(top_10)
				mAPs.append(mAP)

				cur_patience += 1
				if epoch > 0 and top_1 > max_acc_2:
					max_acc_1 = mAP
					max_acc_2 = top_1
					top_5_max = top_5
					top_10_max = top_10

					try:
						best_cluster_info_1[0] = num_cluster
						best_cluster_info_1[1] = outlier_num
					except:
						pass
					cur_patience = 0

					if FLAGS.mode == 'Train':
						if FLAGS.dataset != 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + 'best.ckpt'
						elif FLAGS.dataset == 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + FLAGS.probe_type + '_best.ckpt'
						print(checkpt_file)
						if FLAGS.save_model == '1':
							saver.save(sess, checkpt_file)
				if epoch > 0:
					if dataset == 'CASIA_B':
						print(
							'[MoCos - Probe Evaluation] %s - %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | ' % (
								FLAGS.dataset, FLAGS.probe_type, mAP, max_acc_1, top_1, max_acc_2, top_5, top_5_max, top_10, top_10_max))
					else:
						print(
							'[MoCos - Probe Evaluation] %s - %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | ' % (
								FLAGS.dataset, FLAGS.probe, mAP, max_acc_1, top_1, max_acc_2, top_5, top_5_max, top_10, top_10_max))

					print(
						'Max: %.4f-%.4f-%.4f-%.4f' % (max_acc_1, max_acc_2, top_5_max, top_10_max))
				if cur_patience == patience:
					print(simp_strs)
					break

				def generate_cluster_features(labels, features):
					centers = collections.defaultdict(list)
					for i, label in enumerate(labels):
						if label == -1:
							continue
						centers[labels[i]].append(features[i])

					centers = [
						torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
					]
					centers = torch.stack(centers, dim=0)
					return centers

				ori_train_labels = np.array(ori_train_labels)
				ori_train_labels = np.argmax(ori_train_labels, axis=-1)
				y_true = ori_train_labels
				gt_class_features = generate_cluster_features(ori_train_labels, train_features_all)
				X_train_J_new = X_train_J
				tr_step = 0
				tr_size = X_train_J_new.shape[0]

				mask_rand_save = np.array(mask_rand_save)
				node_mask_save = np.array(node_mask_save)
				batch_SSq_CSP_loss = []
				batch_SSk_CSP_loss = []
				batch_CSP_loss = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					mask_rand = mask_rand_save[tr_step:(tr_step + 1)]
					mask_rand = np.reshape(mask_rand, [time_step])
					mask_rand_2 = mask_rand_save_2[tr_step:(tr_step + 1)]
					mask_rand_2 = np.reshape(mask_rand_2, [time_step])

					gt_labels = y_true[tr_step * batch_size:(tr_step + 1) * batch_size]
					node_mask_rand = node_mask_save[tr_step:(tr_step + 1)]
					node_mask_rand = np.reshape(node_mask_rand, [joint_num])
					if FLAGS.rand_flip == '1':
						sign_flip = np.random.random(pos_enc_ori.shape[1])
						sign_flip[sign_flip >= 0.5] = 1.0
						sign_flip[sign_flip < 0.5] = -1.0
						pos_enc_ori_rand = pos_enc_ori * sign_flip
						_,  CSP_loss_, Seq_features, SSq_CSP_loss_, SSk_CSP_loss_, = sess.run(
							[train_op, CSP_loss, seq_ftr, SSq_CSP_loss, SSk_CSP_loss],
							feed_dict={
								J_in: X_input_J,
								seq_mask: mask_rand,
								seq_mask_2: mask_rand_2,
								node_mask: node_mask_rand,
								L_eig: pos_enc_ori_rand,
								gt_lab: gt_labels,
								gt_class_ftr: gt_class_features,
								train_flag: True
							})
					elif FLAGS.rand_flip == '0':
						_,  CSP_loss_, Seq_features, SSq_CSP_loss_, SSk_CSP_loss_,  = sess.run(
							[train_op, CSP_loss, seq_ftr, SSq_CSP_loss, SSk_CSP_loss],
							feed_dict={
								J_in: X_input_J,
								seq_mask: mask_rand,
								seq_mask_2: mask_rand_2,
								node_mask: node_mask_rand,
								L_eig: pos_enc_ori,
								gt_lab: gt_labels,
								gt_class_ftr: gt_class_features,
								train_flag: True
							})
					if FLAGS.save_flag == '1':
						batch_CSP_loss.append(CSP_loss_)
						batch_SSq_CSP_loss.append(SSq_CSP_loss_)
						batch_SSk_CSP_loss.append(SSk_CSP_loss_)

					if tr_step % display == 0:
						print(
							'[%s] Batch num: %d | CSP Loss: %.5f | SSk-CSP Loss: %.5f | SSq-CSP Loss: %.5f |' %
							(str(epoch), tr_step, CSP_loss_, SSk_CSP_loss_, SSq_CSP_loss_,))
					tr_step += 1
				if FLAGS.save_flag == '1':
					CSP_losses.append(np.mean(batch_CSP_loss))
					SSq_CSP_losses.append(np.mean(batch_SSq_CSP_loss))
					SSk_CSP_losses.append(np.mean(batch_SSk_CSP_loss))

			if FLAGS.save_flag == '1':
				try:
					os.mkdir(pre_dir + dataset + '/' + probe + change + '/')
				except:
					pass
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'top_1s.npy', top_1s)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'top_5s.npy', top_5s)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'top_10s.npy', top_10s)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mAPs.npy', mAPs)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'SSq_CSP_loss.npy', SSq_CSP_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'SSk_CSP_loss.npy', SSk_CSP_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'CSP_loss.npy', CSP_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mACT.npy', mACT)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mRCL.npy', mRCL)

			sess.close()

elif FLAGS.mode == 'Eval':
	checkpt_file = pre_dir + FLAGS.dataset + '/' + FLAGS.probe + change + '/best.ckpt'

	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpt_file + '.meta')

		J_in = loaded_graph.get_tensor_by_name("Input/Placeholder:0")
		L_eig = loaded_graph.get_tensor_by_name("Input/Placeholder_1:0")
		train_flag = loaded_graph.get_tensor_by_name("Input/Placeholder_2:0")
		seq_ftr = loaded_graph.get_tensor_by_name("MoCos/MoCos/Reshape_36:0")

		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		loader.restore(sess, checkpt_file)
		saver = tf.train.Saver()
		mask_rand_save = []
		node_mask_save = []

		def gal_loader(X_train_J, y_train):
			tr_step = 0
			tr_size = X_train_J.shape[0]
			gal_logits_all = []
			gal_labels_all = []
			gal_features_all = []
			embed_1_all = []
			embed_2_all = []

			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
				[Seq_features] = sess.run([seq_ftr],
										  feed_dict={
											  J_in: X_input_J,
											  L_eig: pos_enc_ori,
											  train_flag: False
										  })
				gal_features_all.extend(Seq_features.tolist())
				gal_labels_all.extend(labels.tolist())
				tr_step += 1
			return gal_features_all, gal_labels_all, embed_1_all, embed_2_all

		def evaluation():
			vl_step = 0
			vl_size = X_test_J.shape[0]
			pro_labels_all = []
			pro_features_all = []
			while vl_step * batch_size < vl_size:
				if (vl_step + 1) * batch_size > vl_size:
					break
				X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
				[Seq_features] = sess.run([seq_ftr],
										  feed_dict={
											  J_in: X_input_J,
											  L_eig: pos_enc_ori,
											  train_flag: False
										  })
				pro_labels_all.extend(labels.tolist())
				pro_features_all.extend(Seq_features.tolist())
				vl_step += 1
			X = np.array(gal_features_all)
			y = np.array(gal_labels_all)
			t_X = np.array(pro_features_all)
			t_y = np.array(pro_labels_all)
			t_y = np.argmax(t_y, axis=-1)
			y = np.argmax(y, axis=-1)

			def mean_ap(distmat, query_ids=None, gallery_ids=None,
						query_cams=None, gallery_cams=None):
				# distmat = to_numpy(distmat)
				m, n = distmat.shape
				# Fill up default values
				if query_ids is None:
					query_ids = np.arange(m)
				if gallery_ids is None:
					gallery_ids = np.arange(n)
				if query_cams is None:
					query_cams = np.zeros(m).astype(np.int32)
				if gallery_cams is None:
					gallery_cams = np.ones(n).astype(np.int32)
				# Ensure numpy array
				query_ids = np.asarray(query_ids)
				gallery_ids = np.asarray(gallery_ids)
				query_cams = np.asarray(query_cams)
				gallery_cams = np.asarray(gallery_cams)
				# Sort and find correct matches
				indices = np.argsort(distmat, axis=1)
				matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
				# Compute AP for each query
				aps = []
				if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(1, m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
								 (gallery_cams[indices[i]] != query_cams[i]))

						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						# discard nan
						y_score[np.isnan(y_score)] = 0
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				else:
					for i in range(m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
								 (gallery_cams[indices[i]] != query_cams[i]))
						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				if len(aps) == 0:
					raise RuntimeError("No valid query")
				return np.mean(aps)

			def metrics(X, y, t_X, t_y):
				# compute Euclidean distance
				if dataset != 'CASIA_B':
					a, b = torch.from_numpy(t_X), torch.from_numpy(X)
					m, n = a.size(0), b.size(0)
					a = a.view(m, -1)
					b = b.view(n, -1)
					dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
							 torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
					dist_m.addmm_(1, -2, a, b.t())
					# 1e-12
					dist_m = (dist_m.clamp(min=1e-12)).sqrt()
					mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
					_, dist_sort = dist_m.sort(1)
					dist_sort = dist_sort.numpy()
				else:
					X = np.array(X)
					t_X = np.array(t_X)
					dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_m = np.array(dist_m)
					mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
					dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_sort = np.array(dist_sort)

				top_1 = top_5 = top_10 = 0
				probe_num = dist_sort.shape[0]
				if (FLAGS.probe_type == 'nm.nm' or
						FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(probe_num):
						if t_y[i] in y[dist_sort[i, 1:2]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, 1:6]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, 1:11]]:
							top_10 += 1
				else:
					for i in range(probe_num):
						# print(dist_sort[i, :10])
						if t_y[i] in y[dist_sort[i, :1]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, :5]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, :10]]:
							top_10 += 1
				return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

			mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
			del X, y, t_X, t_y, pro_labels_all, pro_features_all
			gc.collect()
			return mAP, top_1, top_5, top_10


		if dataset == 'KGBD' or dataset == 'KS20':
			# if FLAGS.level == 'J':
			X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
			adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
									   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
									   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

		elif dataset == 'BIWI':
			if probe == 'Walking':
				# if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

			else:
				# if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

		elif dataset == 'IAS':
			if probe == 'A':
				# if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

			else:
				# if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, enc_k=int(FLAGS.enc_k))

		elif dataset == 'CASIA_B':
			X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
			adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
									   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
									   batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[1])
		del _
		gc.collect()

		mAP_max = top_1_max = top_5_max = top_10_max = 0

		gal_features_all, gal_labels_all, gal_embed_1_all, gal_embed_2_all = gal_loader(X_gal_J, y_gal)
		mAP, top_1, top_5, top_10 = evaluation()
		print(
			'[Evaluation on %s - %s] mAP: %.4f | R1: %.4f - R5: %.4f - R10: %.4f |' %
			(FLAGS.dataset, FLAGS.probe, mAP, top_1, top_5, top_10,))
		sess.close()
		exit()

print('----- Model hyperparams -----')
print('f (sequence length): ' + str(time_step))
print('H (embedding size): ' + FLAGS.H)
print('MGT Layers: ' + FLAGS.L_transformer)
print('heads: ' + FLAGS.n_heads)
print('MoCos fusion lambda: ' + FLAGS.fusion_lambda)
print('t1: ' + FLAGS.t_1)
print('t2: ' + FLAGS.t_2)

print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))

print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

print('p_s: ' + FLAGS.prob_s)
print('p_t: ' + FLAGS.prob_t)
print('lambda: ' + FLAGS.fusion_lambda)

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)
