
REM clean up cache and start from scratch
call conda activate base
call conda update --all --yes
call conda clean  --all --yes
call pip cache purge

REM create the full eda environment
call conda activate base
call conda env remove -v -n eda
call conda env create -v -f env_dev_eda.yaml

REM install the full eda environment
call conda activate eda

call pip install --upgrade pip

REM call pip install pipwin
REM call pipwin refresh
REM call pipwin install openexr

call pip install -v -r env_dev_eda.txt
