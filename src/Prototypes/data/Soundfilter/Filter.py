import os
import shutil
import pandas as pd

def copy_folders_matching_labels(csv_path, source_directory, target_directory):
    # Read labels from the CSV file
    print(f"Reading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    labels = df['primary_label'].tolist()

    # Check and create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        print(f"Creating target directory: {target_directory}...")
        os.makedirs(target_directory)

    # Iterate through all folders in the source directory and check if they need to be copied
    print(f"Checking folders in {source_directory}...")
    copied_folders = 0
    for folder_name in os.listdir(source_directory):
        folder_path = os.path.join(source_directory, folder_name)
        if os.path.isdir(folder_path) and folder_name in labels:
            print(f"Copying {folder_name} to {target_directory}...")
            shutil.copytree(folder_path, os.path.join(target_directory, folder_name))
            copied_folders += 1

    print(f"Completed! Copied {copied_folders} folders to {target_directory}.")

# Paths for the directories and CSV
csv_path = "matched_bird_names.csv"
source_directory = "C:\\Users\\22396\\Downloads\\birdclef-2023\\train_audio"
target_directory = "C:\\Users\\22396\\PycharmProjects\\temporaryEcho\\Soundfilter\\Additional birdcall from kaggle"

copy_folders_matching_labels(csv_path, source_directory, target_directory)
