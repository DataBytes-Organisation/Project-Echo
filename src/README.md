## Data Science Development Environment Setup (Windows)

Install Visual Studio Code

Install Miniconda

In c:\ProgramData\pip\pip.ini add these lines:

```
#
# [global]
extra-index-url = https://download.pytorch.org/whl/cu117

# [install]
trusted-host = download.pytorch.org
```

Open a Miniconda prompt

Execute Environment.bat (this will take a while to setup a 'dev' and 'dev_min' and 'scratch' conda environments)

In Visual Studio Code, navigate to File->Open Folder

Select the Project-Echo\src\  directory you have checked out

Create your files e.g. a jupyter notebook file testload.ipynb

Visual Studio Code will automatically detect your file type

Select the 'dev' environment and run your notebook!



For MAC:
Open the src directory in terminal and execute:

```
conda env create -f env_dev.yaml
```

If you do not have anaconda installed it will not work - please ensure anaconda is installed first.
 This doc will help with any issues https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

