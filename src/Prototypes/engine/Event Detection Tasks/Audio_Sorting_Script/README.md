# Audio Processing Toolkit

This toolkit provides two distinct scripts tailored to process and categorize animal sound audio files. Depending on the organization of your dataset, you can select the appropriate script to use.

## 1. Audio Categorization Tool (Event_Segmentation_YamNet01.ipynb)

### Introduction

This script is ideal for scenarios where all audio files are located in a single directory, and there's an accompanying CSV file that maps each audio file to its category.

### Usage

1. Place all your audio files in a single directory.
2. Prepare a CSV file with mappings of audio filenames to their respective scientific names.
3. Modify the paths in the script to point to your audio directory and the CSV file.
4. Run the script. The audio files will be processed and saved in directories named after their scientific names.

## 2. Audio Organization Tool (Event_Segmentation_YamNet02.ipynb)

### Introduction

This script is designed for situations where audio files have already been categorized under subdirectories named in the format "Common Name - Scientific Name". The script will process the audio files and save them under directories named after the scientific names.

### Usage

1. Organize your audio files into subdirectories with the naming convention "Common Name - Scientific Name".
2. Modify the paths in the script to point to your main directory containing the subdirectories.
3. Run the script. The audio files will be processed and re-saved under directories based on their scientific names.

## General Requirements

- Ensure that the required libraries (e.g., TensorFlow, numpy) are installed.
- For optimal performance, consider running the scripts in an environment with a supported GPU.

