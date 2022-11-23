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


-----------------------------------------------------------
For most of these libraries to work - Python 3.9 is required 
On mac, open terminal in this directory and execute the following command:
```
pip install apache-beam
```


```
pip install -r env_dev.txt
```

This will install all the necessary libraries
