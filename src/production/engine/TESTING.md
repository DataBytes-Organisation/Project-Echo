\# Echo Engine Automated Test Instructions



\## Purpose



This test suite provides automated coverage for the core Echo Engine

functions required during Sprint 1. It tests configuration loading,

model loading, audio preprocessing, prediction output and IoT message

handling without requiring the complete dataset or live production

services.



\## Environment



\- Operating system: Windows

\- Python: 3.9.25

\- Pytest: 8.4.2

\- Coverage.py: 7.10.7

\- Conda environment: projectecho



\## Installation



Open Anaconda Prompt and activate the project environment:



```bat

conda activate projectecho

cd /d D:\\Project-Echo\\src\\production\\engine

python -m pip install pytest pytest-cov