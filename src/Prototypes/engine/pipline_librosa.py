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
Input_directory = "C:\\Users\\User\\Documents\\Uni\\2023\\Semester 2\\Project Echo\\test Audio\\in"
Trimmed_directory = "C:\\Users\\User\\Documents\\Uni\\2023\\Semester 2\\Project Echo\\test Audio\\trimmed"
Images_directory = "C:\\Users\\User\\Documents\\Uni\\2023\\Semester 2\\Project Echo\\test Audio\\images"

#Select all of the files in the input directory
for file in os.listdir(Input_directory):

    #Get the extension of the file we are working with
    fileExtension = file.rsplit(".",1)[-1]

    file_path=os.path.join(Input_directory,file)
    #load in audio file to be trimmed
    audio =AudioSegment.from_file(file_path)
    #Time in ms
    start_time=0
    #Time in ms (10s)
    end_time=10000
    #trim the Audio to specifyed time (0-10s) This value can be changed to suit optimal size for training the engine
    trimmed_audio = audio[start_time:end_time]
    output_path=os.path.join(Trimmed_directory,file)
    #save the trimmed audio at the specifyed path
    trimmed_audio.export(output_path, format=fileExtension)

#Create Filter banks for converting Audio to Mel Spectrogram
#Source Valerio Velardo - The Sound of AI Youtube, Valerio Velardo - The Sound of AI 
#(2020) Extracting Mel spectrograms with python, YouTube. Available at: 
#https://www.youtube.com/watch?v=TdnVE5m3o_0 (Accessed: 1 August 2023). 

filter_banks= librosa.filters.mel(n_fft=2048,sr=22050, n_mels=10)
filter_banks.shape
plt.figure(figsize=(25,10))
librosa.display.specshow(filter_banks, sr=sr, x_axis='linear')
plt.colorbar(format="%+2.f")

        
#Select all of the files in the trimmed directory
for file in os.listdir(Trimmed_directory):
    fileName = file.rsplit(".",1)[0]+".png"
    #load audio file with librosa
    file_path=os.path.join(Trimmed_directory,file)
    scale, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=scale, n_fft=2048, hop_length=512, n_mels=50)
    log_mel_spectrogram=librosa.power_to_db(mel_spectrogram)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.f')
    image_dir=os.path.join(Images_directory,fileName)
    plt.savefig(image_dir)
