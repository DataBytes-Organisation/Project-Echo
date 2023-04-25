#!/bin/bash

# Setup WSL2 on windows 11
# Step 0: Open a command prompt
# Step 1: wsl --list --online
# Step 2: wsl --install --distribution Ubuntu-22.04
# Step 3: shutdown /r /t 0
# Step 4: wsl --list --verbose
# Step 5: sudo apt update
# Step 6: sudo apt upgrade -y

# Ensure miniconda is installed
# https://docs.conda.io/en/latest/miniconda.html#linux-installers
rm -rf ~/miniconda3/
rm -f ./Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
chmod +x ./Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
./Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -ub
rm -f ./Miniconda3-py310_23.3.1-0-Linux-x86_64.sh

# ensure bash in initialised with conda env
source ~/miniconda3/etc/profile.d/conda.sh

conda activate base
conda update -y -n base -c defaults conda

# some apt packages
apt-get update
apt-get upgrade -y

# create a dev conda environment
conda activate base
conda env remove -v -n dev
conda create -y -n dev python=3.10 
conda activate dev

# setup conda environment
conda install -y -c conda-forge cudatoolkit=11.8.0

apt-get install -y python3-pip
pip install --upgrade pip

# setup a GPU environment deps via conda
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# create the full dev environment
pip install -r wsl2_ubuntu_requirements.txt

# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

