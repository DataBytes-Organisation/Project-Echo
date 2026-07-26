from pydub import AudioSegment
import os
import random
import csv

# This script efficiently creates mixed audio files by blending human voices with one or two random animal sounds using the pydub, os, random, and csv libraries.
# On execution, it checks for a designated output folder, creating one if missing, then loads a given human voice file.
# From selected subfolders, presumed to be named after animals, it randomly selects WAV files.
# These sounds are combined with interspersed silences, ensuring a minimum length of 10 seconds.
# Details of each generated audio, such as combined sounds and their order, are logged into an audio_info.csv file.
# On completion, the script indicates where the files are stored.
# In essence, it offers a streamlined method to produce and track mixed audio files with varied animal sounds.


def generate_mixed_audio(base_path, human_voice_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the human voice file
    human_voice = AudioSegment.from_wav(human_voice_file)

    # List all the subfolders (animal names)
    animal_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Create a csv file to store the results
    with open(os.path.join(output_folder, "audio_info.csv"), "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Filename", "Sound1", "Sound2", "Sound3"])

        # Generate mixed audio files
        for i in range(1, 101):  # Assuming you want to generate 100 files, if you want 1000, replace this with (1,1001)
            print(f"Generating audio file {i} out of 100...")

            # Decide whether to choose 1 or 2 animal sounds
            num_sounds = random.choice([1, 2])

            # Randomly select animal folders
            selected_folders = random.sample(animal_folders, num_sounds)

            animal_sounds = []
            for folder in selected_folders:
                animal_files = [f for f in os.listdir(os.path.join(base_path, folder)) if f.endswith('.wav')]
                selected_file = random.choice(animal_files)
                sound = AudioSegment.from_wav(os.path.join(base_path, folder, selected_file))
                animal_sounds.append((folder, sound))

            # Decide the lengths for each sound, considering the actual length of the chosen files
            if num_sounds == 1:
                length1 = min(random.randint(2, 7) * 1000, len(animal_sounds[0][1]))
                lengths = [length1]
            else:
                length1 = min(random.randint(2, 4) * 1000, len(animal_sounds[0][1]))
                length2 = min(random.randint(2, min(4, 7 - (length1 / 1000))) * 1000, len(animal_sounds[1][1]))
                lengths = [length1, length2]

            # Create a list of all sounds and shuffle it
            all_sounds = [(human_voice, "human")] + [(sound[1][:lengths[idx]], sound[0]) for idx, sound in
                                                     enumerate(animal_sounds)]
            random.shuffle(all_sounds)

            # Combine all the sounds with random silence intervals between 0.5s to 1s
            combined_sounds = all_sounds[0][0]
            sequence = [all_sounds[0][1]]
            for idx in range(1, len(all_sounds)):
                silence_duration = random.randint(500, 1000)  # Random duration between 0.5s to 1s
                combined_sounds += AudioSegment.silent(duration=silence_duration)
                combined_sounds += all_sounds[idx][0]
                sequence.append(all_sounds[idx][1])

            # Fill with silence if total length is less than 10s
            if len(combined_sounds) < 10000:
                combined_sounds += AudioSegment.silent(duration=(10000 - len(combined_sounds)))

            # Save the combined sound to a file
            combined_sounds.export(os.path.join(output_folder, f"{i}.wav"), format="wav")

            # Write the info to the csv file
            csvwriter.writerow([f"{i}.wav"] + sequence + [""] * (3 - len(sequence)))

    print(f"Generated audio files saved in {output_folder}")


if __name__ == "__main__":
    base_folder_path = r"E:\Training_Data\Test"  # Replace this with your own path
    human_voice_path = r"E:\Training_Data\HumanVoice.wav"  # Replace this with your own path
    output_folder_name = r"E:\Training_Data\MixedAudio"  # Replace this with your own path
    generate_mixed_audio(base_folder_path, human_voice_path, output_folder_name)
