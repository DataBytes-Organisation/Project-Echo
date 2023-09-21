from setuptools import find_packages, setup
setup(
    name='echo_spectogram',
    packages=find_packages(include=['echo_spectogram']),
    version='0.1.0',
    description='Echo Local Python library',
    author='ProjectEcho',
    license='MIT',
    install_requires=['IPython','matplotlib','numpy','librosa','tensorflow==2.10.1','soundfile','audiomentations'],
)