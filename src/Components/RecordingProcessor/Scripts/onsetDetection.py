import os
import numpy as np
import noisereduce as nr
from scipy.io import wavfile
from scipy.signal import spectrogram
from pydub import AudioSegment
from mutagen.wave import WAVE
from mutagen._util import DictProxy
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, lfilter
import taglib


#configure paths
mic = '1'
import_directory = f'../Experiments/0_InitialProcessing/combined'
base_dir = os.path.abspath(os.path.join('..', 'Experiments', '0_InitialProcessing', 'combined'))
export_directory = f'../Experiments/OnsetDetection/combined'

#Ensure export directory exists
if not os.path.exists(export_directory):
    os.makedirs(export_directory)

#Load the audio file


freq_threshold = 2000  #Frequency threshold in Hz for spectrogram analysis
dbThreshold = 30

# load upthe audio file's metadata


#Extract the timestamp from the comments frame metadata from the audio file
#The key 'COMM::\x00\x00\x00' is specific to your metadata structure
#this comment acts as the 'global time' at the time that the recording started

# Define the directory containing the WAV files
directory_path = import_directory

def onsetDetection (spectrogram):
    print("Onset Detection...")

def averageDB ():
    print("Average db level")

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

for file_name in os.listdir(directory_path):
    # Check if the file is a WAV file
    if file_name.endswith('.wav'):
        # Construct the full file path
        file_path = os.path.join(base_dir, file_name)
        audio = AudioSegment.from_wav(file_path)
        # Load the WAV file
        sampling_rate, data = wavfile.read(file_path)
        # Handle stereo audio by averaging the two channels if necessary
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        print(f"!---Processing {file_name} ---!")
        #Apply low-pass filter
        #Cutoff frequency for the low-pass filter
        cutoff_frequency = 12500  
        data_filtered = butter_lowpass_filter(data, cutoff_frequency, sampling_rate)  # Filter out frequencies above 12500 Hz
        
        #Number of points per segment
        #Smaller nperseg = greater time resolution but lower frequency resolution
        #Higher nperseg = lower time resolution but higher frequency resolution
        nperseg = 512  
        noverlap = nperseg * .75

        #The frequency resolution can be calculated as ('Sampling rate'/'nfft')
        #Given our variables where nperseg = 512, cutoff_frequency is 12500, sampling rate = 44100
        #The Frequency resolution will be set to approximately 21.53Hz per bin

        #nfft = int(sampling_rate / (2 * cutoff_frequency) * nperseg)  # Adjust nfft to focus on <= 12500 Hz
        nfft = 2 ** np.ceil(np.log2(sampling_rate / cutoff_frequency * nperseg))

        #Time Resolution = ('nperseg'/'sampling_rate')
        #When nperseg = 512, Sampling Rate = 44100, noverlap hop distance = 128 (nperseg - noverlap)
        #Time resolution for 1 second is approximately 352 time slices, which provides a time resolution of 2.9 milliseconds
        #In this time, sound can only travel .94 meters

        frequencies, times, Sxx = spectrogram(data_filtered, sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Convert Sxx to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        #For experimentation purposes we can standardise the db intensities, such that the maximum measured intensity acts as '0'
        #Standardize dB levels relative to the maximum observed dB level
        #max_dB = np.max(Sxx_dB)
        #Sxx_dB = Sxx_dB - max_dB  # scale so max is at 0 dB

        #Bandwidth variables, lower and upper
        frequencyBenchmark = 4500
        frequencyBenchmark2 = 10000

        #Obtain the indicies of the frequency bins within the desired bandwidth
        freq_indices = np.where((frequencies > frequencyBenchmark) & (frequencies < frequencyBenchmark2))[0]

        #Print out the frequency bins for reference
        #print("Frequency bins within spectrogram:   ",  frequencies)

        #Print how many bins are contained within the bandwidth
        #print("Indices of frequencies between 2500 Hz and 5000 Hz:", freq_indices)

        #Print out the frequencies that are withi the bandwidth
        #print("Frequencies within the specified range:  ", frequencies[freq_indices])

        #Checck and print how many frequency bins there are
        #print("Frequencies length:  ", len(frequencies))

        data_collection = {}

        # Loop through the lists and populate the dictionary
        for i in freq_indices:
            index = i
            frequency = frequencies[i]
            db_values_for_bin = Sxx_dB[i, :]
            median = np.median(db_values_for_bin)
            if abs(median) < 5:
                median = 5
            data_collection[index] = {"frequency": frequency, "median": median}

        #onsetDetection('start')
        #averageDB()
       
        #print(Sxx)
        # Initialize an array to mark time slices where the condition is met
        time_slices_exceeding_threshold = []
        sample = 0
        # Check each time slice

        init = True
        previous = None
        previous2 = None        

        #Onset Detection
        for i, time_slice in enumerate(Sxx_dB.T):
            
            if times[i] > 0.05:
                for frequency_index in freq_indices:
                    median_value = abs(data_collection[frequency_index]['median'])
                    if init and (time_slice[frequency_index] > median_value * 2.5) and (previous[frequency_index] > median_value * 2.5) and (previous2[frequency_index] > median_value *2.5):
                        onsetTime = times[i-2]  # This should capture the first valid time
                        init = False  # Ensure this is updated only once
                        print(f"Vocalization point of interest found at time: {onsetTime}")
                        print(f"Frequency:  {frequencies[frequency_index]}")
                        print(f"Detected:   {time_slice[frequency_index]}")
                        print(f"Median: {data_collection[frequency_index]['median']}")

                if np.any(time_slice[freq_indices] > dbThreshold):
                    #print(f'Exeeded: {i}',time_slice[freq_indices])
                    # Mark the time slice if the condition is met
                    time_slices_exceeding_threshold.append(times[i])
                    
            previous2 = previous
            previous = time_slice
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')
        plt.title(f'Spectrogram with Threshold Notches for {file_name}')

        # Plot red notches along the x-axis line where dB level exceeds the threshold

        #for time_slice in time_slices_exceeding_threshold:
        #    plt.plot([time_slice, time_slice], [0, max(frequencies)], 'r-', linewidth=0)  # Vertical line
        #    plt.plot(time_slice, 0, 'r|', markersize=0.1)  # Notch
        if init == False:

            plt.axvline(x=onsetTime, color='red', linewidth=1)  # Vertical line
            plt.scatter(onsetTime, 0, color='red', marker='|', s=100)  # Notch at the bottom

            #Renaming the filename
            parts = file_name.split('_')

            if len(parts) == 2:  # This checks if the split result is exactly two parts (id and number.wav)
                id_part = parts[0]
                number_part = parts[1].split('.')[0]  # Extract the number and ignore the extension

                # Here you can modify the number part as needed
                # Example: Increment the number by 1
                new_number_part = str(int(number_part) + (onsetTime*1000))  # Increment the number
                new_number_part = round(float(new_number_part))
                print(f"Renaming with timestamp of {new_number_part}.")

                # Construct the new filename
                new_file_name = f"{id_part}_{new_number_part}.wav"
                new_file_path = os.path.join(base_dir, new_file_name)
                
                # Rename the file
                if not os.path.exists(new_file_path):

                    os.rename(file_path, new_file_path)
                    print(f'Renamed "{file_name}" to "{new_file_name}"')
            
                    print("File Path    ",file_path)

                    #Adding metadata to the audio files
                    audioData = taglib.File(f'{new_file_path}')

                    if audioData.tags:
                        print("Audio data tags found...")

                    else:
                        print("No audio data tags found...")
                        audioData.tags['ARTIST'] = f'{id_part}'
                        audioData.tags['COMMENT'] = f'{onsetTime}'
                        audioData.save()
                        audioData.close()

                #print("Audio Data:  ", audioData.tags)
            
        # Set x-axis limit to ensure notches are visible at the edge of the plot
        plt.xlim([min(times), max(times)])

        if init == False:
        # Save the plot as a JPEG image
            output_file_name = f'./{export_directory}/Single Onset{os.path.splitext(new_file_name)[0]}.jpg'

        else:
           output_file_name = f'./{export_directory}/No Detection{os.path.splitext(file_name)[0]}.jpg'

        plt.savefig(output_file_name, dpi=300)
        plt.close()
        #print("Count", count)
        if init == True:
            print("---couldn't find onset---")
        print(f"!---Processed {file_name} ---!")
        print()
        print()
        sample += 1