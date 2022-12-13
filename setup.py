import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="echo",
    py_modules=["echo"],
    version="1.0",
    description="Robust Bioaccoustic Recognition and Classification Tool",
    readme="README.md",
    python_requires=">=3.7",
    author="Deakin University 2022 T3 Project Echo DataByte Capstone A Team",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
