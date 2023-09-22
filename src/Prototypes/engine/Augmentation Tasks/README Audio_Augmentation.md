# Augmentations for Audio in Animal Noise Classification

## Introduction

Welcome to the Augmentations for Audio in Animal Noise Classification code repository! This repository provides a set of augmentation techniques for audio data, particularly tailored for animal noise classification tasks. These augmentations aim to enhance the diversity and robustness of your audio dataset, helping your model better handle variations in animal sounds and real-world environmental noise.

## Augmentation Techniques

### Time Stretch
Time stretching changes the duration of an audio clip while preserving its pitch. This augmentation can help the model become more robust to variations in the speed of animal sounds, which may occur due to factors like proximity or age.

### Pitch Shift
Pitch shifting alters the pitch (frequency) of an audio clip while maintaining its duration. This augmentation can be useful for capturing variations in animal sounds caused by factors like age, gender, or individual differences in pitch.

### Add Background Noise
Adding background noise can help the model generalize better to real-world scenarios where animal sounds are often accompanied by environmental noise. It can also make the model more robust to noise interference during inference.

### Time Mask
Time masking involves replacing a portion of the audio with silence or zero values. This can help the model focus on specific segments of animal sounds that are more discriminative while ignoring irrelevant or noisy parts.

### Seven Band Parametric EQ
Applying an equalization filter with seven bands can enhance or suppress certain frequency components in the audio. This augmentation can be beneficial when the distinctive features of animal sounds lie in specific frequency bands.

### Padding
Padding involves extending the duration of audio clips by adding silence to the beginning or end. Padding can help the model handle varying input lengths and improve its robustness to short or long animal calls.

### Add Gaussian SNR
Adding Gaussian noise with a specific Signal-to-Noise Ratio (SNR) can simulate different levels of noise in the environment. This augmentation helps the model learn to distinguish animal sounds from background noise at various SNR levels.

### Add Short Noises
Adding short, random noise bursts to the audio can help the model become more robust to sudden environmental disturbances. It simulates situations where animal calls might be interrupted by other sounds.

## Code Usage

### Random Augmentations

In this code repository, you can apply these augmentations to your audio files. Here's how to use the provided code:

1. Define the paths to your background noise file, short noise files or directory, data directory containing your original audio files, and the output directory where augmented files will be saved.

BACKGROUND_NOISE_PATH = "/path/to/background_noise.wav"  # Path to your background noise file
SHORT_NOISE_PATH = "/path/to/short_noise.wav"  # Path to short noise/noises directory
data_directory = "/path/to/your/audio/data/directory"  # Path to your data directory
output_directory = "/path/to/your/output/data/directory"  # Path to your output directory

2. Use the `random_augmentations` function to create a composition of the mentioned augmentations randomly selected with a probability of 0.1, and then apply them to your audio files in the specified data directory.

transform = random_augmentations()

### Augmentations

This code repository also offers the flexibility to apply specific augmentations of your choice to your audio files. Here's how to use it:

1. Define the paths to your data directory containing the original audio files and the output directory where augmented files will be saved.

BACKGROUND_NOISE_PATH = "/path/to/background_noise.wav"  # Path to your background noise file
SHORT_NOISE_PATH = "/path/to/short_noise.wav"  # Path to short noise/noises directory
data_directory = "/path/to/your/audio/data/directory"  # Path to your data directory
output_directory = "/path/to/your/output/data/directory"  # Path to your output directory

2. Customize the `aug_mapping` dictionary to specify which augmentations you want to apply and their corresponding indices.

aug_mapping = {
    0: 'TimeStretch',
    1: 'PitchShift',
    2: 'BackgroundNoise',
    3: 'ShortNoises',
    4: 'TimeMask',
    5: 'SevenBandParametricEQ',
    6: 'Padding',
    7: 'AddGaussianSNR'
}

3. Use the provided code to apply the selected augmentations to your audio files.

transform = index_to_transformation(index)

### Implementation 1

To execute the procedure, it's necessary to define the paths for both our data and output directories. Subsequently, subfolders for each augmentation are established within the output directory, wherein the newly augmented files are stored in a format identical to that found in the data directory.

Output directory structure: `/path/Augmentation_Name/species_name/audio_files.wav`

### Implementation 2

To execute the procedure, it's necessary to define the paths for both our data and output directories. Afterward, the newly augmented files are stored in a format identical to that found in the data directory, with the augmented name added as a prefix to the audio file name.

Output directory structure: `/path/species_name/augmentation_name_audio_files.wav`

### Display Audio and Spectrograms based on implementation 2

You can visualize the effects of these augmentations on your audio files by using the provided functions to plot waveforms and mel spectrograms. Additionally, you can listen to the augmented audio for qualitative assessment.

## Conclusion

These audio augmentations are valuable tools for enhancing your dataset and improving the performance of your animal noise classification model. Feel free to customize and extend the code to suit your specific requirements and explore further enhancements. Happy coding!