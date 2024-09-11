import os
import taglib
import time

# Specify the directory path
directory_path = '../Experiments/0_InitialProcessing/combined'

# List all files and directories in the specified path
files_and_directories = os.listdir(directory_path)
print(files_and_directories)

# If you only want files and not directories, you can filter the list
only_files = [f for f in files_and_directories if os.path.isfile(os.path.join(directory_path, f))]
print(only_files)

detections = {}
keys = []

for file_name in only_files:
    parts = file_name.split('_')
    if len(parts) == 2:  # This checks if the split result is exactly two parts (id and number.wav)
        id_part = parts[0]
        number_part = parts[1].split('.')[0]  # Extract the number and ignore the extension
        #print(f"Renaming with timestamp of {number_part}.")
        
        try:
            meta = taglib.File(f'{directory_path}\{file_name}')
            print("Time of vocalization in audio clip:  ",  meta.tags['COMMENT'][0])
            print("Microphone: ", meta.tags['ARTIST'][0])
            print("Global Time of detection:    ", number_part)

            #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if f'{id_part}'not in detections:
                print('###############################################')
                print("Not in dictionary")
                print("Create new dictionary entry")
                detections[f'{id_part}'] = []
                detections[f'{id_part}'].append(int(number_part))
                keys.append(f'{id_part}')

            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                detections[f'{id_part}'].append(int(number_part))
        except Exception:
            print("No metadata found in .wav file, continuing to next file...")
            continue

for mic in detections:
    print("Printing entry:   ",mic)
    for time in mic:
        print("Time:    ",time)

#timerange 300 milliseconds, sound can travel 100 meters in 291 ms, as the 9 milliseconds as a buffer
timeRange = 1500
clustered = []

for key in keys: 
    for entry in detections[f'{key}']:
        event = []
        event.append(entry)
        #print(f"Mic: {key}  Time:   {entry}")
        for key2 in keys:
            if key2 == key:
                continue
            for entry2 in detections[f'{key2}']:
                if abs(entry - entry2) < timeRange:
                    print("Potential linked event found")
                    print(f"Entry x: {key}_{entry} Entry y: {key2}_{entry2}")
                    event.append(entry2)
                    detections[f'{key2}'].remove(entry2)
                    break
        clustered.append(event)
        detections[f'{key}'].remove(entry)

count1 = 0
count2 = 0
count3 = 0

for event in clustered:
    if len(event) == 1:
        count1 += 1
    elif len(event) == 2:
        count2 += 1
    else:
        count3 += 1
        print("Event:   ",event)

print(f"Amount of 1 mic events:  {count1}")
print(f"Amount of 2 mic events:  {count2}")
print(f"Amount of 3 mic events  {count3}")

