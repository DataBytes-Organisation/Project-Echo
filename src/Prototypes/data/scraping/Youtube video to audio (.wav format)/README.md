# Engine - Random Sampling

## Overview 
The code provides a function, random_move_files, designed to reorganize files across directories. Given an input directory with species subfolders, the function checks the directory's validity, creates a matching structure in an output directory, and then, for each species folder, randomly selects and moves a default of three files to the corresponding folder in the output directory. Care should be taken when using this function as files are moved, not copied, potentially leading to data loss if not backed up.

## Details :

# Directory Verification: 
It first checks if the given input directory exists. If the output directory doesn't exist, it creates one.

# Folder Iteration: 
# For each species folder present in the input directory:
1. The function ensures that the path being considered is indeed a folder.
2. If a corresponding folder doesn't exist in the output directory, one is created.
3. It then retrieves a list of all the files present in that species folder.

# Random File Selection and Movement: 
# From the list of files in the current species folder:
1. A subset of files (default is 3) is randomly selected.
2. These selected files are then moved from the input directory to the corresponding folder in the output directory.

#### Note: This code is designed to move files (not copy), so it's crucial to have backups of the data before executing the function to prevent accidental data loss.

## Importance 
This code plays a crucila role in selecting random audio files which is beneficial for training.

## Input 

https://drive.google.com/file/d/1yJ1VPbq9NeDyHtnyExKOS2tCr2R47bgg/view?usp=sharing

## Output 

https://drive.google.com/file/d/1qBQRaUaGghws0PhraXe33cYOwHP5jFGY/view?usp=sharing

Access this link to see the sampled audio.
