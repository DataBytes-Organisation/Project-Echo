# Echo Spectrogram Python Library

Welcome to the Echo Spectrogram Python Library! This library provides a set of functions to work with audio data, including plotting waveforms and mel spectrograms. It is a part of the Echo Local project.

# How to Add a New Function

Add the function in the echo_spectrogram directory and make sure to update the __init__ file.

Update setup.py to add required dependencies. Add them in install_requires.

## Run the setup.py file to build the wheel (whl) file
python setup.py bdist_wheel

The wheel file would be created in the dist folder.

## Run the uninstall command to remove if the package is already installed
pip uninstall dist/echo_spectrogram-0.1.0-py3-none-any.whl

## Finally, install the updated package
pip install dist/echo_spectrogram-0.1.0-py3-none-any.whl

# How to Use the Library

from echo_spectrogram import DisplayWave, DisplayMelSpec, show_example_images, display_img_after_aug, apply_augmentation

## To plot waveform
DisplayWave("/path/to/your/audio_file.wav", True, True, True, "This is the Image Title")

## To plot MelSpectrogram
DisplayMelSpec("/path/to/your/audio_file.wav", True, True, True, True, True, "This is the Image Title")

## To display Show Example Image
show_example_images("/path/to/your/audio/directory", 2)

## To Show Images after applying augmentations
output_path = "/path/to/your/output/directory"
input_path = "/path/to/your/audio_file.wav"
display_img_after_aug(input_path, output_path)

## Apply Single Augmentation
output_path = "/path/to/your/output/directory"
input_path = "/path/to/your/audio_file.wav"
apply_augmentation(input_path, output_path, AddGaussianNoise, 0.001, 0.015)