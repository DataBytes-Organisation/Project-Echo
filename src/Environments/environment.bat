
REM clean up cache and start from scratch
call conda activate base
call conda update --all --yes
call conda clean  --all --yes
call pip cache purge

REM create the full dev environment
call conda activate base
call conda env remove -v -n dev
call conda env create -v -f env_dev_win.yaml

REM install the full dev environment
call conda activate dev

call pip install --upgrade pip

call pip install pipwin
call pipwin refresh
call pipwin install openexr

call pip install -v -r env_dev_win.txt
