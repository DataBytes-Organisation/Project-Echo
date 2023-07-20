
# Environments Setups

This README provides a little background on environment setups.

The files in this folder represent a good starting point, but need to be customised for your component.

The is an attempt to cover linux, mac and windows environments.

## Windows 11

Use the Environment.bat file to build your conda environment - execute from within the directory.

```
Project-Echo\src\Environments> environment.bat
```

## Mac

Use the env_dev_mac.yaml file to build your conda environment.

```
conda env create -f env_dev_mac.yaml
```

## Linux

On windows, there is a Windows Sub-system for Linux 2 (WSL2) which provides a linux option.

To get started, enable WSL and create your Ubuntu 22.04 Linux image from the windows command prompt

```
wsl --install Ubuntu-22.04
```
 
Create a data science / AI environment by installing Lambda Stack:

```
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | sh -
sudo reboot
```

Add a few extra items from APT which are needed:

```
sudo apt-get install -y python3-pip 
sudo apt-get install -y libopenexr-dev 
sudo apt-get install -y python-is-python3
```

install python dependencies:

```
pip install -c constraints_wsl2.txt -r env_dev_wsl2.txt
```
