# Environments Setups

This README provides a little background on environment setups.

The files in this folder represent a good starting point, but need to be customised for your component.

The is an attempt to cover linux, mac and windows environments.

## Windows 11

Likewise for Mac, installing and using Anaconda Navigator may be the better option. Follow the links below, or search to install Anaconda for Windows.

Use the Environment.bat file to build your conda environment - execute from within the directory.

```
Project-Echo\src\Environments> environment.bat
```

## Mac

> _Instructions for Trimester 2 2023_

_Once a virtual environment is set up, always launch Jupyter Notebook from this environment._

#### Using Anaconda Navigator:

You can use Anaconda Navigator to create a new environment. This is a much easier to switch environments, access the environment's terminal and launch Jupyter Notebook. Name it _ProjectEcho_. In this case, ignore the below code and create the environment directly from _Anaconda Navigator_. For help check out:  
[Installing Anaconda](https://docs.anaconda.com/free/navigator/install/)  
[Creating Environments in Anaconda](https://docs.anaconda.com/free/navigator/tutorials/manage-environments/) (create new env with python 3.8)

Once the environment is created, launch terminal from this environment. The list of libraries to install are included in the text file below and currently include all libraries that have been used up until now for developing and testing. Some of them may no longer be required.

From `~\Project-Echo\src\Environments` run the following command to install the dependencies and libraries:

```
pip install -r env_dev_mac.txt
```
after this, install tensorflow : 
Run the following command:

`conda install -y -c apple tensorflow-deps`
`python -m pip install tensorflow-macos tensorflow-metal`



<br/>

#### Using command line:

There are possible errors with this, so if it doesnt work, try the above option. Use the `env_dev_mac.yaml` file to build your conda environment.  
\_Note this file has not been updated, so may require additional libraries. Check out the text file to find any missing.\*

From within `~\Project-Echo\src\Environments` run:

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

Install python dependencies:

```
pip install -c constraints_wsl2.txt -r env_dev_wsl2.txt
```
