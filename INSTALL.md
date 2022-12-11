## Project Echo Development Environment Setup

### Windows Users

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

### For MAC Users:

Open the src directory in terminal and execute:

```
conda env create -f env_dev_mac.yaml
```

If you do not have anaconda installed it will not work - please ensure anaconda is installed first.
 This doc will help with any issues https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

