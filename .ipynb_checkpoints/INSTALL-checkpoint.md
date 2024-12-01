## Project Echo Development Environment Setup

### Windows Users

The following steps should get your started:

Install Visual Studio 2022 Community Edition [https://visualstudio.microsoft.com/vs/community/] with c++ modules installed

Install Visual Studio Code [https://code.visualstudio.com/]

Install Miniconda [https://docs.conda.io/en/latest/miniconda.html]

In c:\ProgramData\pip\pip.ini add these lines:

```
#
# [global]
extra-index-url = https://download.pytorch.org/whl/cu117

# [install]
trusted-host = download.pytorch.org
```

Check the Project Echo code from github [https://github.com/stephankokkas/Project-Echo]

Open a Miniconda prompt via the start menu

Execute src/Environment.bat (this will take a while to setup a 'dev' conda python based development environment)

In Visual Studio Code, navigate to File->Open Folder

Select the Project-Echo directory you have just checked out above

Create your files e.g. a jupyter notebook file testload.ipynb

Visual Studio Code will automatically detect your file type

Select the 'dev' environment (top right of notebook) and run your notebook!

### MAC Users

Open the src directory in terminal and execute:

```
conda env create -f env_dev_mac.yaml
```

If you do not have anaconda installed it will not work - please ensure anaconda is installed first.
 This doc will help with any issues https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


### Linux Users

No instructions at this time.




# Project Echo

Project echo is a bioaccouostic classification tool.


## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1, but the codebase is expected to be compatible with Python 3.7 or later and recent PyTorch versions. The codebase also depends on a few Python packages like [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) for reading audio files. The following command will pull and install the latest commit from this repository, along with its Python dependencies 

    pip install git+https://github.com/stephankokkas/Project-Echo/tree/SK-PIP-Package-with-new-main/project-echo-package.git

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/stephankokkas/Project-Echo/tree/SK-PIP-Package-with-new-main/project-echo-package.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Command-line usage

Not yet implemented


## Python usage

Transcription can also be performed within Python: 

```python
import project_echo

model = echo.load_model("base")
result = model.predict("audio.mp3")
```
