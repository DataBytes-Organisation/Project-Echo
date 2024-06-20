import time
import pytz
import os
import taglib
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from mutagen.wave import WAVE
from mutagen._util import DictProxy
import datetime
from scipy.signal import spectrogram, butter, lfilter
import requests


directory_path = '../Experiments/0_InitialProcessing/combinedCopy'
url = "http://localhost:9000/volume/upload_wav"

files_and_directories = os.listdir(directory_path)
base_dir = os.path.abspath(os.path.join('..', 'Experiments', '0_InitialProcessing', 'combinedCopy'))
only_files = [f for f in files_and_directories if os.path.isfile(os.path.join(base_dir, f))]
simulationStartTime = int(time.time()*1000)
#print(only_files)

# print(files_and_directories)
# print(base_dir)
# print(only_files)
# for file in only_files:
#     file = os.path.join(base_dir,file)
    #print(file)

def sortSecond(val):
    return val[1]

def simulated_stopwatch(start_time_millis, duration_seconds, timeJump):
    end_time_millis = int((time.time() + duration_seconds) * 1000)
    current_time_millis = start_time_millis
    
    try:
        count = 0
        detections = 0
        times = []
        for file_name in only_files:
                parts = file_name.split('_')
                mic = parts[0]
                startTime = int(parts[1].split('.')[0])
                time_list = (mic, startTime, file_name)
                times.append(time_list)
            
        times.sort(key=sortSecond)
        #print(times)
        
        print("Current time millis: ",current_time_millis)
        print("End time millis: ",end_time_millis)
        while current_time_millis <= end_time_millis:
            pop = 0
            #print(len(only_files))
            
            for sorted_file in times:
                file = sorted_file[2]
                startTime = int(sorted_file[1])+timeJump
                mic = sorted_file[0]
                #print(f"Mic: {mic}   Time:   {startTime}")
                #print(directory_path,file)
                # print(base_dir,file)
                #print(current_time_millis-startTime)
                if current_time_millis - startTime > 4000:
                    
                    print(f"File found...{file}")
                    epoch_time = (startTime+timeJump)/1000
                    utc_datetime = datetime.datetime.fromtimestamp(epoch_time, datetime.timezone.utc)

                    # Convert the UTC datetime to AEST
                    aest_timezone = pytz.timezone('Australia/Sydney')
                    aest_datetime = utc_datetime.astimezone(aest_timezone)

                    # Print the result
                    print(aest_datetime.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
                    #print(file)
                    #print(current_time_millis - startTime)
                    if file.endswith('.wav'):
                        filePath = os.path.join(directory_path,file)
                        with open(filePath, 'rb') as fl:
                            # Define the files dictionary with content type
                            files = {'file': (file, fl, 'audio/wave')}
                            new_file_name = f"{mic}_{startTime}.wav"
                            data = {'new_file_name' : new_file_name}
                            # Make the POST request with the file in the body
                            response = requests.post(url, files=files, data = data)
                            print(f"Popping: {file}")
                            print(response)
                            print("")
                            times = times[1:]
                            detections +=1
                else:
                    continue
            count += 1
            #current_time_millis += 1
            current_time_millis = int(time.time()*1000)
            #time.sleep(0.001)  # Sleep for 1 millisecond
            
            if count % 100000 == 0:
            
                print(f"Total time simulation has been running (s): {(int(time.time()*1000)-simulationStartTime) / 1000}")
                print(f"Total detects fed through:  {detections}")

    except KeyboardInterrupt:
        print("Stopping the simulated stopwatch.")
if __name__ == "__main__":

    timeNow = int(time.time() * 1000)

    # Set a fixed start time in milliseconds (for example, current time)
    fixed_start_time = 1710389866820
    timeJump = timeNow - fixed_start_time
    print("Time jump:   ", timeJump)
    # Duration for the simulation in seconds (e.g., 10 seconds)
    simulation_duration = 1200
    simulated_stopwatch(fixed_start_time, simulation_duration, timeJump)
