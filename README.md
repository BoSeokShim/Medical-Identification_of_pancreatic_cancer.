### Convolutional neural network model for automatic recognitionand classification of pancreatic cancer cell based on analysis of lipid droplet on unlabeled sample by 3D optical diffraction tomography

> Computer Methods and Programs in Biomedicine, 2024. (IF: 6.1, JCR 13.1%)

# Contents
1. [Overview](#Overview)
2. [Introduction](#Introduction)
3. [Dataset composition](#Dataset-composition)
4. [Main results](#Main-Results)
5. [Setup](#Setup)
6. [Getting started](#Getting-started)
7. [Training](#Training)
    - [Single Task Setting]
    - [Multi-view Task Setting]
    - [Multi-Task Setting]
    - [Modal-Task Setting(meta-data)]
    - [Final Task Setting(multi-view, multi-task, meta-data)]
    
8. [Evaluating](#Evaluating)
    - [Single Task Setting]
    - [Multi-view Task Setting]
    - [Multi-Task Setting]
    - [Modal-Task Setting(meta-data)]
    - [Final Task Setting(multi-view, multi-task, meta-data)]
9.  [Acknowledgments](#Acknowledgments)
10.  [Citation](#Citation)

# Overview
![스크린샷 2024-02-05 171501](https://github.com/tmdrn9/Medical-Identification_of_pancreatic_cancer/assets/77779116/26649b96-d580-48da-9e79-62919cc48ae2)

> This study proposed an automatic pancreatic cancer cell recognition system utilizing a deep convolutional neural network and quantitative images of lipid droplets (LDs) from stain-free cytologic samples through optical diffraction tomography. We retrieved 3D refractive index tomograms, reconstructing 37 optical images per cell. Additionally, we employed various machine learning techniques within a single image-based prediction model to enhance the computer-aided diagnostic system's performance.

# Introduction
> Pancreatic cancer cells generally accumulate large numbers of lipid droplets (LDs), which regulate lipid storage. To promote rapid diagnosis, an automatic pancreatic cancer cell recognition system based on a deep convolutional neural network was proposed in this study using quantitative images of LDs from stain-free cytologic samples by optical diffraction tomography.
# Dataset composition

    ${ROOT}
     `-- cell
         |-- cancer
         |    |-- BxPc-3
         |    |    `-- cell ID
         |    |           |-- Num.1 cell
         |    |           |     |-- image
         |    |           |     |     |-- BxPc-3-001_1.jpg
         |    |           |     |     |-- BxPc-3-001_2.jpg
         |    |           |     |     |-- BxPc-3-001_3.jpg
         |    |           |     |     |-- ...
         |    |           |     |     `-- BxPc-3-001_37.jpg
         |    |           |     |
         |    |           |     `-- meta data
         |    |           |           |-- Cell volume         
         |    |           |           |-- Cell surface area      
         |    |           |           |-- Projected area        
         |    |           |           |-- Mean RI         
         |    |           |           |-- Protein concentration        
         |    |           |           |-- Dry mass
         |    |           |           |-- Sphericity
         |    |           |           |-- Lipid droplet volume
         |    |           |           |-- Lipid droplet count
         |    |           |           `-- ...         
         |    |           |                  
         |    |           |-- Num.2 cell
         |    |           |-- Num.3 cell
         |    |           `-- Num.n cell
         |    |-- Capan-2
         |    `-- PSN-1
         |
         `-- normal
              `-- H6c7

# Main results
### Results on Ablation study
||AUC|Accuracy|F1-score|Precision|Recall|
|:----------|----|----|----|----|----|
|Baseline|0.985|94.39|0.955|0.969|0.942|
|Modal-Task|0.987|95.52|0.965|0.972|0.958|
|Multi-view Task|0.995|97.59|0.981|0.988|0.974|
|Multi-view & Modal Task|0.996|96|0.987|0.985|0.988|
|Final Task|0.998|96.77|0.988|0.992|0.985|
### Note:
- Input Image size is 256x256
- Backbone network is EfficientNet-b3

# Setup
- Python 3.8
- CUDA Version 12.4
- cuDNN Version 8.9.7

1. Nvidia driver, Anaconda install

2. Install pytorch

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. Install various necessary packages in requirements.txt

        pip install -r requirements.txt
   
# Getting started
When using Terminal, directly execute the code below after setting the path

        python train.py --kernel-type baseline --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16

When using pycharm:

        Menu Run 
        -> Edit Configuration 
        -> Check train.py in Script path
        -> Go to parameters and enter the following

        --kernel-type baseline --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16

        -> Running/debugging after check Apply button

As training progresses, the best and final weights are saved in the folder `./weights/`. Learning logs are saved in the `./logs/` folder.

# Training
## Single Task Setting

    python train.py --kernel-type baseline --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16

## Multi-view Task Setting

    python train.py --kernel-type grouping --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --GROUPING

## Multi-Task Setting

    python train_multitask.py --kernel-type meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 

## Modal-Task Setting

     python train.py --kernel-type grouping_meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta --GROUPING

## Final Task Setting

    python train_multitask.py --kernel-type meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta --GROUPING

    
# Evaluating

## Single Task Setting

    python evaluate.py --kernel-type baseline --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16

## Multi-view Task Setting

    python evaluate.py --kernel-type grouping --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --GROUPING

## Multi-Task Setting

    python evaluate_multitask.py --kernel-type meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 

## Modal-Task Setting

     python evaluate.py --kernel-type grouping_meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta --GROUPING

## Final Task Setting

    python evaluate_multitask.py --kernel-type meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta --GROUPING

# Acknowledgments

> This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT: Ministry of Science and ICT) (No. 2022R1C1C1006242).

# Citation

> If you want to cite our paper and code, you can use a bibtex code here:

        @article{hong2024convolutional,
          title={Convolutional neural network model for automatic recognition and classification of pancreatic cancer cell based on analysis of lipid droplet on unlabeled sample by 3D optical diffraction tomography},
          author={Hong, Seok Jin and Hou, Jong-Uk and Chung, Moon Jae and Kang, Sung Hun and Shim, Bo-Seok and Lee, Seung-Lee and Park, Da Hae and Choi, Anna and Oh, Jae Yeon and Lee, Kyong Joo and others},
          journal={Computer Methods and Programs in Biomedicine},
          volume={246},
          pages={108041},
          year={2024},
          publisher={Elsevier}
        }
