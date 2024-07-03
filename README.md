# Structural Tensor and Frequency Guided Semi-Supervised Segmentation for Medical Images
by Xuesong Leng, Xiaxia Wang, Wenbo Yue, Jianxiu Jin, Guoping Xu

## Installation
Ubuntu 22.04
Conda Environment Setup
Create your own conda environment

conda create -n STFA python=3.10
conda activate STFA

Install Pytorch == 2.0.0(depends on your NVIDIA driver and you can see your compatible CUDA version at the right hand corner in nvidia-smi)

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

python required packages

pip install -r requirements.txt

## Run

train command


test command