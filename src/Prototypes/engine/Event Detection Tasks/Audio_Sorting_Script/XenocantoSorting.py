
from core import *

import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import wave

import librosa
import numpy as np
import pandas as pd
import params
import yamnet
# Load YAMNet
yamnet_model = yamnet.yamnet_frames_model(params)
yamnet_model.load_weights('yamnet.h5')
yamnet_classes = yamnet.class_names('yamnet_class_map.csv')

# Set constants for audio processing
SC = {
    'AUDIO_SAMPLE_RATE': 48000,
    'MODEL_INPUT_IMAGE_WIDTH': 260,
    'MODEL_INPUT_IMAGE_HEIGHT': 260,
    'MODEL_INPUT_IMAGE_CHANNELS': 3,
}

# Function to split a single audio file
def split_audio(input_file, output_dir, min_dur=1, max_dur=2, max_silence=0.5, energy_threshold=58):
    audio_regions = split(
        input_file,
        min_dur=min_dur,
        max_dur=max_dur,
        max_silence=max_silence,
        energy_threshold=energy_threshold,
        sw=2
    )

    for i, r in enumerate(audio_regions):
        # Regions returned by `split` have 'start' and 'end' metadata fields
        print(f"Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s")

        # Use YAMNet to recognize audio
        should_save = recognize_audio(r)

        if should_save:
            # Specify the full path to save the region in the output directory
            filename = os.path.join(output_dir, f"region_{r.meta.start:.3f}-{r.meta.end:.3f}.wav")
            r.save(filename)
            print(f"Region saved as: {filename}")

# Function to preprocess and recognize audio using YAMNet
def recognize_audio(audio_region):
    waveform = audio_region.samples  # Extract waveform directly
    if waveform.ndim == 2 and waveform.shape[0] == 2:
        waveform = waveform[1, :]  # Take one channel if it's stereo

    # Perform mono conversion if needed
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=0)

    # Ensure that the sample width is set to 2 bytes (16-bit)
    sample_width = 2

    waveform = waveform.astype(np.int16)  # Set the sample width to 2 bytes
    waveform = waveform.astype(np.float32) / 32768.0  # Normalize to the range [-1, 1]

    prediction = np.mean(yamnet_model.predict(np.reshape(waveform, [1, -1]), steps=1)[0], axis=0)
    sorted_indices = np.argsort(prediction)[::-1]

    matching_classes = set(yamnet_classes[sorted_indices]).intersection(set(['Dog', 'Cat', 'Bird', 'Animal', 'Birdsong', 'Canidae', 'Feline', 'Livestock', 'Rodents, Mice', 'Wild animals', 'Pets', 'Frogs', 'Insect', 'Snake', 'Domestic animals, pets', 'crow']))
    is_condition_met = any([prediction[idx] > 0.2 for idx in sorted_indices if yamnet_classes[idx] in matching_classes])

    return is_condition_met


def main():
    # Directory containing your audio files
    folder_path = r"E:\Training_Data\Vulpes vulpes"
    # Output directory for storing the segmented files
    output_dir = r"E:\Training_Data\Vulpes"

    # List to store the names of problematic files
    problematic_files = []

    # Loop through all subdirectories in the main folder
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            # Extract the scientific name from the directory name
            _, scientific_name = dir_name.split(' - ', 1)

            # Create a subfolder for the scientific name if it doesn't exist
            scientific_folder = os.path.join(output_dir, scientific_name)
            os.makedirs(scientific_folder, exist_ok=True)

            # Directory path of the current subfolder
            subfolder_path = os.path.join(root, dir_name)

            # Loop through all audio files in the subfolder
            for _, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith(".mp3") or file.endswith(".wav"):
                        # Split the audio and save it in the corresponding subfolder
                        file_path = os.path.join(subfolder_path, file)
                        try:
                            split_audio(file_path, scientific_folder)
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
                            problematic_files.append(file_path)

    # Print the list of problematic files at the end
    if problematic_files:
        print("\nFiles that could not be processed:")
        for file in problematic_files:
            print(file)


if __name__ == "__main__":
    main()

