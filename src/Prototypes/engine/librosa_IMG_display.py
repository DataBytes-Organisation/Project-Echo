# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import librosa, librosa.display
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from IPython.display import Audio

n_mels=256
hop_length=128
fmax=8000



AUDIO_CLIP_DURATION = 5 # seconds 
AUDIO_NFFT =  2048
AUDIO_WINDOW = None
AUDIO_STRIDE = 200
AUDIO_SAMPLE_RATE = 48000
AUDIO_MELS = 260
AUDIO_FMIN = 20
AUDIO_FMAX =13000
AUDIO_TOP_DB = 80

Trimmed_directory = "D:\\Downloads\\Data\\wind\\trimmed"

def DisplayWave(filepath,plt_x_axis,plt_y_axis,plt_title,image_title):
    audio, sr = librosa.load(filepath)
    plt.figure(figsize=(20, 10))
    librosa.display.waveshow(y=audio, sr=sr)
    if plt_title:
        plt.title(image_title, fontdict=dict(size=18))
    if plt_x_axis:    
        plt.xlabel('Time', fontdict=dict(size=15))
    if plt_y_axis:
        plt.ylabel('Amplitude', fontdict=dict(size=15))
    plt.show()
    print("Success")
    
def DisplayMelSpec(filepath,plt_x_axis,plt_y_axis,plt_title,plt_colourBar,image_title):
    if plt_x_axis:
        x_axis_label ='time'
    else:
        x_axis_label =None
        
    if plt_y_axis:
        y_axis_label ='mel'
    else:
        y_axis_label =None   
    
    audio, sr = librosa.load(filepath)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,n_mels=AUDIO_MELS, hop_length=AUDIO_STRIDE, fmax=AUDIO_FMAX)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #plt.title("Log Mel Spectrogram of Audio")    
    img = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),x_axis=x_axis_label,  y_axis=y_axis_label, hop_length = 128, fmax=8000)
    Plot_colourBar=plt_colourBar
    if Plot_colourBar:
        fig.colorbar(img, ax=ax, format=f'%0.2f dB')
    if plt_title:
        plt.title(image_title)
    
    plt.show()
    

DisplayWave('D:\\Downloads\\Data\\wind\\trimmed\\wind686064__lewisounds__cold-wind-blowing-outside.mp4', True,True,True,"Title of the image")
#DisplayMelSpec('D:\\Downloads\\Data\\wind\\trimmed\\wind4018__noisecollector__synthwind.wav', True,True,True,True,"Title of the image")