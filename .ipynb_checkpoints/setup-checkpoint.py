from setuptools import setup, find_packages

setup(
    name="Echo",
    py_modules=["Echo"],
    version="1.0",
    description="Robust Bioaccoustic Recognition and Classification Tool",
    readme="README.md",
    python_requires=">=3.7",
    author="Deakin University 2022 T3 Project Echo DataByte Capstone A Team",
    packages=find_packages(include=['Echo', 'Echo.*']),
    install_requires=[
        'numpy>=1.21.0,<1.25.0',
        'ffmpeg-python==0.2.0',
        'tensorflow==2.10.0',  #Versions Fixed
        'tensorflow-io==0.31.0',  #Versions Fixed
        'keras==2.10.0',  #Versions Fixed
        'pydub>=0.25.1',
        'scipy>=1.7.0',
        'numba>=0.56.0'
    ],
    package_data={
        'Echo': [
            'Models/echo_model/1/saved_model.pb',
            'Models/echo_model/1/variables/*',
            'Models/echo_model/1/assets/*'
        ],
    },
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)