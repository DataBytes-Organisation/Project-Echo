"""
******************************************************************************************
* ⚠️Note: This script is tailored for the optimized_weather_training_pipeline.ipynb only *
******************************************************************************************
"""
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

SC = {
    'AUDIO_CLIP_DURATION': 2,
    'AUDIO_NFFT': 2048,
    'AUDIO_WINDOW': None,
    'AUDIO_STRIDE': 200,
    'AUDIO_SAMPLE_RATE': 16000,
    'AUDIO_MELS': 260,
    'AUDIO_FMIN': 20,
    'AUDIO_FMAX': 13000,
    'AUDIO_TOP_DB': 80,
}

def load_and_pad_audio(file_path, duration=2, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    required_samples = sr * duration
    if len(audio) < required_samples:
        audio = np.pad(audio, (0, required_samples - len(audio)), 'constant')
    else:
        audio = audio[:required_samples]
    return audio, sr

def prepare_audio(file_path):
    audio, sr = load_and_pad_audio(file_path, duration=SC['AUDIO_CLIP_DURATION'], sr=SC['AUDIO_SAMPLE_RATE'])
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=SC['AUDIO_NFFT'],
        hop_length=SC['AUDIO_STRIDE'],
        n_mels=SC['AUDIO_MELS'],
        fmin=SC['AUDIO_FMIN'],
        fmax=SC['AUDIO_FMAX']
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, top_db=SC['AUDIO_TOP_DB'])
    spectrogram_resized = tf.image.resize(log_mel_spectrogram[np.newaxis, :, :, np.newaxis], [260, 260])
    spectrogram_resized = np.repeat(spectrogram_resized, 3, axis=-1)
    return spectrogram_resized


file_path = r"C:\Users\22396\Downloads\rain sample.wav"

model = load_model('WeatherAudioDetectionModel.h5')

input_spectrogram = prepare_audio(file_path)

predictions = model.predict(input_spectrogram)

predicted_class = np.argmax(predictions, axis=1)

if predicted_class.size == 1:
    predicted_class = predicted_class.item()

encoder = LabelEncoder()
encoder.fit(['Earthquake', 'Rain', 'Thunder', 'Tornado'])
predicted_label = encoder.inverse_transform([predicted_class])

print("Predicted class index:", predicted_class)
print("Predicted class label:", predicted_label[0])

print("All class probabilities:", predictions.flatten())

max_probability = np.max(predictions)
print("Highest class probability:", max_probability)

confidence_threshold = 0.7
if max_probability > confidence_threshold:
    print("High confidence prediction:", predicted_label[0])
else:
    print("Low confidence prediction: need further review")
