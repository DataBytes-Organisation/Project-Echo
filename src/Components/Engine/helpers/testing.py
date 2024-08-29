import melspectrogram_to_cam
import numpy as np
import tensorflow as tf
import librosa

# Generate a sample Mel spectrogram
y, sr = librosa.load(librosa.example('trumpet'))
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert to decibel scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Convert to tensor and add batch dimension
mel_spectrogram_tensor = tf.expand_dims(tf.convert_to_tensor(mel_spectrogram_db), -1)
mel_spectrogram_tensor = tf.repeat(mel_spectrogram_tensor, 3, axis=-1)
mel_spectrogram_tensor = tf.expand_dims(mel_spectrogram_tensor, 0)

# Call the convert function
overlayed_image = melspectrogram_to_cam.convert(mel_spectrogram_tensor)

# Display or save the result
import matplotlib.pyplot as plt
plt.imshow(overlayed_image)
plt.title("CAM Overlay on Mel Spectrogram")
plt.show()