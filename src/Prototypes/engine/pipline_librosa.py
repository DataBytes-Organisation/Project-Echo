# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:39:33 2023

@author: User
"""

import os
import shutil
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
from pydub import AudioSegment

#Specify input and output directories
Input_directory = "D:\\Downloads\\Data\\wind\\in"
Trimmed_directory = "D:\\Downloads\\Data\\wind\\trimmed"
Images_directory = "C:\\Users\\User\\Documents\\Uni\\2023\\Semester 2\\Project Echo\\test Audio\\images\\wind"

#Create Filter banks for converting Audio to Mel Spectrogram
#Source Valerio Velardo - The Sound of AI Youtube, Valerio Velardo - The Sound of AI 
#(2020) Extracting Mel spectrograms with python, YouTube. Available at: 
#https://www.youtube.com/watch?v=TdnVE5m3o_0 (Accessed: 1 August 2023). 

print("Creating filter banks")
filter_banks= librosa.filters.mel(n_fft=2048,sr=22050, n_mels=10)
filter_banks.shape
plt.figure(figsize=(25,10))
librosa.display.specshow(filter_banks, x_axis='linear')
plt.colorbar(format="%+2.f")

        
#Select all of the files in the trimmed directory
for file in os.listdir(Trimmed_directory):
    try:
        fileName = file.rsplit(".",1)[0]+".png"
        #load audio file with librosa
        file_path=os.path.join(Trimmed_directory,file)
        scale, sr = librosa.load(file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=scale, n_fft=2048, hop_length=512, n_mels=50)
        log_mel_spectrogram=librosa.power_to_db(mel_spectrogram)
        plt.figure(figsize=(25,10))
        librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.f dB')
        image_dir=os.path.join(Images_directory,fileName)
        plt.savefig(image_dir)
        plt.close()
    except:
        print("Unable to convert file")
