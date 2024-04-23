import os
import numpy as np
import noisereduce as nr
from scipy.io import wavfile
from scipy.signal import spectrogram
from pydub import AudioSegment
from mutagen.wave import WAVE
from datetime import datetime
import matplotlib.pyplot as plt

#The current state of this processsing script.
#1. Analyze the audio file, create chunks lasting 4 seconds of audio starting at a point where the loudness crosses a threshold
#2. Create and analyze the spectragrams of each  audio chunk. Keep any chunk where the decible level crosses a desired threshold above a certain frequency.
#3. Output the spectragrams,2 will be created, one as is, another with red notches to show where the conditions in step #2 were satisfied.

#To do
#1. Begin to determine vocalization start time, right now the audio file names 
#1. Incorporate pitch modulation detection to improve vocalization start time 

#configure paths
input_audio_path = 'recording1Mic1.wav'
export_directory = 'mic4'

#Ensure export directory exists
if not os.path.exists(export_directory):
    os.makedirs(export_directory)

#Load the audio file
audio = AudioSegment.from_wav(input_audio_path)

# Parameters
sample_rate = audio.frame_rate
loudness_threshold = -15  #dBFS
segment_length_ms = 4000  #4 seconds in milliseconds
check_interval_ms = 0.05 #Check every sample


freq_threshold = 2000  #Frequency threshold in Hz for spectrogram analysis
dbThreshold = 30

# load upthe audio file's metadata
audioData = WAVE(input_audio_path)

#Extract the timestamp from the comments frame metadata from the audio file
#The key 'COMM::\x00\x00\x00' is specific to your metadata structure
#this comment acts as the 'global time' at the time that the recording started

if 'COMM::\x00\x00\x00' in audioData.tags:
    #get the timestamp that is stored as the first (and likely only) item in the 'text' list of the COMM frame
    timestamp_ms_str = audioData.tags['COMM::\x00\x00\x00'].text[0]
    timestamp_ms = int(timestamp_ms_str)
    print("GLOBAL STARTING TIME:    ",timestamp_ms)
    # Convert milliseconds to seconds
    timestamp_s = timestamp_ms / 1000.0

    # Convert to a datetime object
    dt_object = datetime.fromtimestamp(timestamp_s)

    print("Date and Time:", dt_object)
else:
    print("Timestamp metadata not found.")

#start time for audio chunks
start_time = 0
i = 0

# Convert check_interval_ms to samples
check_interval_samples = int(check_interval_ms * audio.frame_rate / 1000)


while start_time < len(audio):
    try:
        segment_end = start_time + segment_length_ms
        startNew = start_time
        if start_time > 100:
            segment = audio[startNew-100:segment_end]
        else:
            segment = audio[startNew:segment_end]
        # Convert PyDub segment to samples for FFT and spectrogram
        samples = np.array(segment.get_array_of_samples())

        if segment.channels > 1:
            samples = samples.reshape(-1, segment.channels)
            samples = np.mean(samples, axis=1)  # Averaging channels for mono

        # Perform spectrogram analysis
        frequencies, times, Sxx = spectrogram(samples, sample_rate)
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        freq_indices = np.where(frequencies > freq_threshold)[0]

        #Check if any of the frequencies exceed the dB threshold
        exceeds_threshold = np.any(Sxx_dB[freq_indices, :] > dbThreshold, axis=0)

        if np.any(exceeds_threshold):
            print("Threshold Crossed. Seconds into recording:  ", start_time/1000)
            print("Check_interval_samples:  ", check_interval_samples)
            # If any frequency components exceed the dB threshold, export the segment
            filename = f"{start_time+timestamp_ms}_chunk_{i}.wav"
            filepath = os.path.join(export_directory, filename)
            segment.export(filepath, format="wav")
            print(f"Exported {filename}")
            i += 1
            start_time += segment_length_ms  # Move to the end of the current segment
        else:
            start_time += check_interval_samples  # Move to the next check interval
            
    except Exception:
        print(Exception)

    

print("Finished exporting chunks.")

# Define the directory containing the WAV files
directory_path = f'./{export_directory}'

# Iterate over each file in the directory
for file_name in os.listdir(directory_path):
    # Check if the file is a WAV file
    if file_name.endswith('.wav'):
        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)
        
        # Load the WAV file
        sampling_rate, data = wavfile.read(file_path)
        
        # Handle stereo audio by averaging the two channels if necessary
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)

        # Compute the spectrogram
        nperseg = min(256, len(data))
        frequencies, times, Sxx = spectrogram(data, sampling_rate, nperseg=nperseg)

        # Convert Sxx to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        # Find indices of frequencies over 2.5 kHz
        frequencyBenchmark = 2500
        freq_indices = np.where(frequencies > frequencyBenchmark)[0]

        # Initialize an array to mark time slices where the condition is met
        time_slices_exceeding_threshold = []

        # Check each time slice
        for i, time_slice in enumerate(Sxx_dB.T):  # Transpose to iterate over time slices
            if np.any(time_slice[freq_indices] > dbThreshold):
                # Mark the time slice if the condition is met
                time_slices_exceeding_threshold.append(times[i])

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')
        plt.title(f'Spectrogram with Threshold Notches for {file_name}')

        # Plot red notches along the x-axis line where dB level exceeds the threshold
        for time_slice in time_slices_exceeding_threshold:
            plt.plot([time_slice, time_slice], [0, max(frequencies)], 'r-', linewidth=.5)  # Vertical line
            plt.plot(time_slice, 0, 'r|', markersize=10)  # Notch

        # Set x-axis limit to ensure notches are visible at the edge of the plot
        plt.xlim([min(times), max(times)])

        # Save the plot as a JPEG image
        output_file_name = f'./{export_directory}/spectrogram_Notches{os.path.splitext(file_name)[0]}.jpg'
        plt.savefig(output_file_name, dpi=300)
        plt.close()

        print(f"Processed {file_name}.")

for file_name in os.listdir(directory_path):
    # Check if the file is a WAV file
    if file_name.endswith('.wav'):
        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)
        
        # Load the WAV file
        sampling_rate, data = wavfile.read(file_path)
        
        # Handle stereo audio by averaging the two channels if necessary
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)

        # Compute the spectrogram
        nperseg = min(256, len(data))
        frequencies, times, Sxx = spectrogram(data, sampling_rate, nperseg=nperseg)

        # Convert Sxx to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        # Find indices of frequencies over 2 kHz
        frequencyBenchmark = 2000
        freq_indices = np.where(frequencies > frequencyBenchmark)[0]

        # Initialize an array to mark time slices where the condition is met
        time_slices_exceeding_threshold = []
        dbThreshold = 20  # dB threshold

        #plotting the spectrogram
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')
        plt.title(f'Spectrogram with Threshold Notches for {file_name}')

        # Set x-axis limit to ensure notches are visible at the edge of the plot
        plt.xlim([min(times), max(times)])

        #save as jpeg
        output_file_name = f'./{export_directory}/spectrogram_' + os.path.splitext(file_name)[0] + '.jpg'
        plt.savefig(output_file_name, dpi=300)
        plt.close()

        print(f"Processed {file_name}.")