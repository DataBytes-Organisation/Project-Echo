# Echo Spectrogram Python Library

Welcome to the Echo Spectrogram Python Library! This library provides a set of functions to work with audio data, including plotting waveforms and mel spectrograms. It is a part of the Echo Local project.

# How to Add new function

Add the function in the echo_spectogram directory and make sure to update the __init__ file

Update setup.py to add required dependencies. Add them in install_requires.

## Run the setup.py file to build the whell (whl) file
python setup.py bdist_wheel

The wheel file would be created in the dist folder

## Run the unistall command to remove if the package is already installed
pip uninstall dist/echo_spectogram-0.1.0-py3-none-any.whl

## Finnally install the updated package
pip install dist/echo_spectogram-0.1.0-py3-none-any.whl

# How to use the library

from echo_spectogram import plot_waveform, plot_mel_spectrogram

## To plot waveform
plot_waveform('audio_filepath')

## To plot MelSpectogram
plot_mel_spectrogram('audio_filepath')