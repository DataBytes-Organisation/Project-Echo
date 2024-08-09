#This script is responsible for comparing two segments of audio and producing a score based on how similar they are
#The idea for this is such that it could potentially be used as a component for pattern detection where an animal (mainly bird)
#produces a vocalization that contains repeating 'syllables'.

import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# Load two audio files
audio_path1 = './3_3.wav'
audio_path2 = './6.wav'
y1, sr1 = librosa.load(audio_path1)
y2, sr2 = librosa.load(audio_path2)

# Ensure that the sample rates are the same
if sr1 != sr2:
    raise ValueError("Sample rates of the two audio files must match")

# Bandpass filter to isolate the frequency range (example: 300Hz to 8000Hz)
from scipy.signal import butter, lfilter

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y

# Apply the bandpass filter
lowcut = 2500.0
highcut = 5000.0
y1_filtered = bandpass_filter(y1, lowcut, highcut, sr1)
y2_filtered = bandpass_filter(y2, lowcut, highcut, sr2)

# Extract MFCCs from the filtered audio
mfcc1 = librosa.feature.mfcc(y=y1_filtered, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2_filtered, sr=sr2, n_mfcc=13)

# Compute the mean of the MFCCs over time
mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

# Standardize the MFCCs
scaler = StandardScaler()
mfcc1_scaled = scaler.fit_transform(mfcc1_mean.reshape(-1, 1)).flatten()
mfcc2_scaled = scaler.fit_transform(mfcc2_mean.reshape(-1, 1)).flatten()

# Compute a similarity score (Euclidean distance in this case)
similarity_score = euclidean(mfcc1_scaled, mfcc2_scaled)

print(f"The similarity score between the audio segments is: {similarity_score}")