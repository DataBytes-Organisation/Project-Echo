import warnings
warnings.filterwarnings('ignore')

import librosa, librosa.display
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from IPython.display import Audio

# these are global constants in the engine pipeline and throughout echonet. Stick to these for all spectrogram generations
AUDIO_CLIP_DURATION = 5 # seconds 
AUDIO_NFFT =  2048
AUDIO_WINDOW = None
AUDIO_STRIDE = 200
AUDIO_SAMPLE_RATE = 48000
AUDIO_MELS = 260
AUDIO_FMIN = 20
AUDIO_FMAX =13000
AUDIO_TOP_DB = 80


def plot_waveform(file_path):
    y, sr = librosa.load(file_path)
    Audio(file_path)
    plt.figure(figsize=(20, 10))
    librosa.display.waveshow(y, sr=sr)

    plt.title('Waveplot', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Amplitude', fontdict=dict(size=15))
    plt.show()




def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=AUDIO_MELS, hop_length=AUDIO_STRIDE, fmax=AUDIO_FMAX)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.title('Mel Spectrogram')
    plt.show()