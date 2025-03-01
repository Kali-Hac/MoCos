![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![Tensorflow >=1.14.0](https://img.shields.io/badge/Tensorflow->=1.14.0-yellow.svg)
![Pytorch >=1.1.0](https://img.shields.io/badge/Pytorch->=1.1.0-green.svg)
![Faiss-gpu >= 1.6.3](https://img.shields.io/badge/Faiss->=1.6.3-orange.svg)

# Motif Guided Graph Transformer with Combinatorial Skeleton Prototype Learning for Skeleton-Based Person Re-Identification
By Haocong Rao and Chunyan Miao. In AAAI 2025 ([**Arxiv**](https://arxiv.org/abs/2412.09044)),

## Introduction
This is the official implementation of MoCos presented by "Motif Guided Graph Transformer with Combinatorial Skeleton Prototype Learning for Skeleton-Based Person Re-Identification". The codes are used to reproduce experimental results of the proposed TranSG framework in the paper.

![image](https://github.com/Kali-Hac/MoCos/blob/main/img/overview.png)
**Abstract**: 

Person re-identification (re-ID) via 3D skeleton data is a challenging task with significant value in many scenarios. Existing skeleton-based methods typically assume virtual motion relations between all joints, and adopt average joint or sequence representations for learning. However, they rarely explore key body structure and motion such as gait to focus on more important body joints or limbs, while lacking the ability to fully mine valuable spatial-temporal sub-patterns of skeletons to enhance model learning. This paper presents a generic Motif guided graph transformer with Combinatorial skeleton prototype learning (MoCos) that exploits *structure-specific* and *gait-related* body relations as well as combinatorial features of skeleton graphs to learn effective skeleton representations for person re-ID. In particular, motivated by the locality within joints' structure and the body-component collaboration in gait, we first propose the *motif guided graph transformer (MGT)* that incorporates hierarchical structural motifs and gait collaborative motifs, which simultaneously focuses on multi-order local joint correlations and key cooperative body parts to enhance skeleton relation learning. Then, we devise the *combinatorial skeleton prototype learning (CSP)* that leverages random spatial-temporal combinations of joint nodes and skeleton graphs to generate diverse *sub-skeleton* and *sub-tracklet* representations, which are contrasted with the most representative features (*prototypes*) of each identity to learn class-related semantics and discriminative skeleton representations. Extensive experiments validate the superior performance of MoCos over existing state-of-the-art models. We further show its generality under RGB-estimated skeletons, different graph modeling, and unsupervised scenarios.


## Environment
- Python >= 3.5
- Tensorflow-gpu >= 1.14.0
- Pytorch >= 1.1.0
- Faiss-gpu >= 1.6.3

Here we provide a configuration file to install the extra requirements (if needed):
```bash
conda install --file requirements.txt
```

**Note**: This file will not install tensorflow/tensorflow-gpu, faiss-gpu, pytroch/torch, please install them according to the cuda version of your graphic cards: [**Tensorflow**](https://www.tensorflow.org/install/pip), [**Pytorch**](https://pytorch.org/get-started/locally/). Take cuda 9.0 for example:
```bash
conda install faiss-gpu cuda90 -c pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install tensorflow==1.14
conda install scikit-learn
```

## Datasets and Models
We provide three already **pre-processed datasets** (IAS-Lab, BIWI, KGBD) with various sequence lengths (**f=4/6/8/10/12**) [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg) and the **pre-trained models** [**here (pwd: weww)**](https://pan.baidu.com/s/1DZtYtLVAbhZPtJ58POackg). Since we report the average performance of our approach on all datasets, here the provided models may produce better results than the paper. <br/>

Please download the pre-processed datasets and model files while unzipping them to ``Datasets/`` and ``ReID_Models/`` folders in the current directory. <br/>

**Note**: The access to the Vislab Multi-view KS20 dataset and large-scale RGB-based gait dataset CASIA-B are available upon request. If you have signed the license agreement and been granted the right to use them, please email us with the signed agreement and we will share the complete pre-processed KS20 and CASIA-B data. The original datasets can be downloaded here: [IAS-Lab](http://robotics.dei.unipd.it/reid/index.php/downloads), [BIWI](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20](http://vislab.isr.ist.utl.pt/datasets/#ks20), [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp). We also provide the ``Data-process.py`` for directly transforming original datasets to the formatted training and testing data. <br/> 

## Dataset Pre-Processing
To (1) extract 3D skeleton sequences of length **f=6** from original datasets and (2) process them in a unified format (``.npy``) for the model inputs, please simply run the following command: 
```bash
python Data-process.py 6
```
**Note**: If you hope to preprocess manually (or *you can get the [already preprocessed data (pwd: 7je2)](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg)*), please frist download and unzip the original datasets to the current directory with following folder structure:
```bash
[Current Directory]
├─ BIWI
│    ├─ Testing
│    │    ├─ Still
│    │    └─ Walking
│    └─ Training
├─ IAS
│    ├─ TestingA
│    ├─ TestingB
│    └─ Training
├─ KGBD
│    └─ kinect gait raw dataset
└─ KS20
     ├─ frontal
     ├─ left_diagonal
     ├─ left_lateral
     ├─ right_diagonal
     └─ right_lateral
```
After dataset preprocessing, the auto-generated folder structure of datasets is as follows:
```bash
Datasets
├─ BIWI
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ Still
│      │    └─ Walking
│      └─ train_npy_data
├─ IAS
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ A
│      │    └─ B
│      └─ train_npy_data
├─ KGBD
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ gallery
│      │    └─ probe
│      └─ train_npy_data
└─ KS20
    └─ 6
      ├─ test_npy_data
      │    ├─ gallery
      │    └─ probe
      └─ train_npy_data
```
**Note**: KS20 data need first transforming ".mat" to ".txt". If you are interested in the complete preprocessing of KS20 and CASIA-B, please contact us and we will share. We recommend to directly download the preprocessed data [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg).

## Model Usage

To (1) train MoCos to obtain skeleton representations and (2) validate their effectiveness on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python MoCos.py --dataset KS20 --probe probe

# Default options: --dataset KS20 --probe probe --length 6  --gpu 0
# --dataset [IAS, KS20, BIWI, KGBD]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# --length [4, 6, 8, 10, 12] 
# --(H, n_heads, L_transfomer, fusion_lambda, prob_s, prob_t, lr, etc.) with default settings for each dataset
# --mode [Train (for training), Eval (for testing)]
# --gpu [0, 1, ...]

```
Please see ```MoCos.py``` for more details.

To print evaluation results (Top-1, Top-5, Top-10 Accuracy, mAP) of the best model saved in default directory (```ReID_Models/(Dataset)/(Probe)```), run:

```bash
python MoCos.py --dataset KS20 --probe probe --mode Eval
```


## Application to Model-Estimated Skeleton Data 

### Estimate 3D Skeletons from RGB-Based Scenes
To apply our MoCos to person re-ID under the large-scale RGB scenes (CASIA B), we exploit pose estimation methods to extract 3D skeletons from RGB videos of CASIA B as follows:
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human body joints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Estimate the 3D human body joints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)


We provide already pre-processed skeleton data of CASIA B for **single-condition** (Nm-Nm, Cl-Cl, Bg-Bg) and **cross-condition evaluation** (Cl-Nm, Bg-Nm) (**f=40/50/60**) [**here (pwd: 07id)**](https://pan.baidu.com/s/1_Licrunki68r7F3EWQwYng). 
Please download the pre-processed datasets into the directory ``Datasets/``. <br/>

### Usage
To (1) train the MoCos to obtain skeleton representations and (2) validate their effectiveness on the person re-ID task on CASIA B under **single-condition** and **cross-condition** settings, please simply run the following command:

```bash
python MoCos.py --dataset CAISA_B --probe_type nm.nm --length 40

# --length [40, 50, 60] 
# --probe_type ['nm.nm' (for 'Nm' probe and 'Nm' gallery), 'cl.cl', 'bg.bg', 'cl.nm' (for 'Cl' probe and 'Nm' gallery), 'bg.nm']  
# --(H, n_heads, L_transfomer, fusion_lambda, prob_s, prob_t, lr, etc.) with default settings
# --gpu [0, 1, ...]

```

Please see ```MoCos.py``` for more details.

## Citation
If you find our work useful for your research, please cite our paper
```bash
@inproceedings{rao2025motif,
  title     = {Motif Guided Graph Transformer with Combinatorial Skeleton Prototype Learning for Skeleton-Based Person Re-Identification},
  author    = {Rao, Haocong and Miao, Chunyan},
  booktitle = {Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2025}
}
```
A professionally curated list of resources (paper, code, data, etc.) on 3D Skeleton Based Person Re-ID (SRID) is available at [SRID Survey](https://github.com/Kali-Hac/SRID).

More awesome skeleton-based models are collected in our [Awesome-Skeleton-Based-Models](https://github.com/Kali-Hac/Awesome-Skeleton-Based-Models).

## License

MoCos is released under the MIT License. Our models and codes must only be used for the purpose of research.

