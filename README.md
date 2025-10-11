#The name of the program:Few-Shot Object Detection in Remote Sensing Imagery via Multi-Scale Feature Fusion


#The title of the manuscript:Few-Shot Object Detection in Remote Sensing Imagery via Multi-Scale Feature Fusion


#The author details:Bo Wang1，Jun Yang1, 2


1. Faculty of Geomatics, Lanzhou Jiaotong University, Lanzhou 730070

3. School of Electronic and Information Engineering, Lanzhou Jiaotong University, Lanzhou 730070



# Multi-Scale-Feature-Fusion
An adaptive dense fusion pyramid enhances cross-scale interaction and feature integrity, deformable convolution improves localization of irregular objects, and a cross-level fused RoI extractor strengthens focus on both targets and context.
# OpenMMLab Few-Shot Setup

This repository contains the setup instructions for running MMFewShot with different GPU models.  
Please follow the steps below **based on your GPU type**.

---

## 🚀 Installation Guide

### 1. Create Conda Environment
```bash
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
```
### 2. Install PyTorch
For NVIDIA RTX 3090 (CUDA 11.1):
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
For NVIDIA RTX 3090 (CUDA 11.1):
```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```
### 3. Install OpenMIM
```bash
pip install openmim
```
### 4. Install Required Packages
```bash
pip install dataclasses future
mim install mmcv==1.4.0
mim install mmcls==0.15.0
mim install mmdet==2.20.0
mim install mmcv-full==1.4.0
```
### 5. Clone and Install MMFewShot
```bash
git clone https://github.com/open-mmlab/mmfewshot.git
cd mmfewshot
pip install -r requirements/build.txt
pip install -v -e .
```
### 6. Fix YAPF Version (if error occurs)
If you see the error:
```css
TypeError: FormatCode() got an unexpected keyword argument 'verify'
```
Please run:
```bash
pip uninstall yapf
pip install yapf==0.40.1
```
### Next, download the installation package from this project and extract it.  
After extraction, navigate to the `mmfewshot` directory and add the files `ADFN.py` , `PH-ROI.py` and `RPN.py`.

## 📂 Dataset Placement

Please download the required datasets (e.g., NWPU VHR-10.v2, DIOR) and place them under the `data` directory as follows:

```text
mmfewshot
├── mmfewshot
├── tools
├── configs
├── data/
│   ├── NWPU/
│   │   ├── JPEGImages/
│   │   ├── ImageSets/
│   │   ├── seed1/
│   │   └── Annotations/
│   └── DIOR/
│   │   ├── JPEGImages/
│   │   ├── ImageSets/
│   │   ├── seed1/
│   │   └── Annotations/
└── ...


## 🚀 train

base train
```bash
bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/nwpu/split1/resnet_B/meta-rcnn_r101_c4_8xb4_nwpu-split1_base-training.py 1
```
few-tine
```bash
bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/nwpu/split1/resnet_B/meta-rcnn_r101_c4_8xb4_nwpu-split1_1shot-fine-tuning.py 1

bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/nwpu/split1/resnet_B/meta-rcnn_r101_c4_8xb4_nwpu-split1_2shot-fine-tuning.py 1

bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/nwpu/split1/resnet_B/meta-rcnn_r101_c4_8xb4_nwpu-split1_3shot-fine-tuning.py 1

bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/nwpu/split1/resnet_B/meta-rcnn_r101_c4_8xb4_nwpu-split1_5shot-fine-tuning.py 1

bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/nwpu/split1/resnet_B/meta-rcnn_r101_c4_8xb4_nwpu-split1_10shot-fine-tuning.py 1
```
### Should you have any questions or suggestions related to this project, please do not hesitate to reach out to us at qwe1459836106@163.com.
