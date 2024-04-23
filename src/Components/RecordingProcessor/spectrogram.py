import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import os

# Define the directory containing the WAV files
directory_path = './cleaned'

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

        # Find indices of frequencies over 2 kHz
        frequencyBenchmark = 2000
        freq_indices = np.where(frequencies > frequencyBenchmark)[0]

        # Initialize an array to mark time slices where the condition is met
        time_slices_exceeding_threshold = []
        dbThreshold = 20  # dB threshold

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
        output_file_name = 'spectrogram_with_notches_' + os.path.splitext(file_name)[0] + '.jpg'
        plt.savefig(output_file_name, dpi=300)
        plt.close()

        print(f"Processed {file_name}.")