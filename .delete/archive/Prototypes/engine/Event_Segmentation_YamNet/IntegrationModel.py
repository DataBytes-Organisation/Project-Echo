import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

import pyaudio
import wave

import librosa
import numpy as np
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import params
import yamnet

# Set constants for audio processing
SC = {
    'AUDIO_CLIP_DURATION': 5,  # seconds
    'AUDIO_NFFT': 2048,
    'AUDIO_WINDOW': None,
    'AUDIO_STRIDE': 200,
    'AUDIO_SAMPLE_RATE': 48000,
    'AUDIO_MELS': 260,
    'AUDIO_FMIN': 20,
    'AUDIO_FMAX': 13000,
    'AUDIO_TOP_DB': 80,

    'MODEL_INPUT_IMAGE_WIDTH': 260,
    'MODEL_INPUT_IMAGE_HEIGHT': 260,
    'MODEL_INPUT_IMAGE_CHANNELS': 3,
}

# Load YAMNet
yamnet_model = yamnet.yamnet_frames_model(params)
yamnet_model.load_weights('yamnet.h5')
yamnet_classes = yamnet.class_names('yamnet_class_map.csv')

#Load specialized engine
model = tf.keras.models.load_model('Replace this with your own model path.')
df = pd.read_csv('Classes.csv')
class_names = df["Class Name"].tolist()

# Using YamNet to recognize audio for further classification
def recognize_audio(waveform, expected_class_names, top_n=10):
    prediction = np.mean(yamnet_model.predict(np.reshape(waveform, [1, -1]), steps=1)[0], axis=0)
    sorted_indices = np.argsort(prediction)[-top_n:][::-1]

    matching_classes = set(yamnet_classes[sorted_indices]).intersection(set(expected_class_names))
    is_condition_met = any(
        [prediction[idx] > 0.2 for idx in sorted_indices if yamnet_classes[idx] in matching_classes])

    if is_condition_met:
        return True
    return False

# Function to preprocess audio sample
def process_audio(waveform):
    # Load and decode the audio file
    # tmp_audio_t = None
    # with open(audio_filepath, 'rb') as file:
    #     tmp_audio_t, _ = librosa.load(file, sr=SC['AUDIO_SAMPLE_RATE'])
    #     if tmp_audio_t.ndim == 2 and tmp_audio_t.shape[0] == 2:
    #         tmp_audio_t = tmp_audio_t[1, :]
    #     tmp_audio_t = tmp_audio_t.astype(np.float32)
    # sample = tmp_audio_t

    # Audio clip function.
    duration_secs = SC['AUDIO_CLIP_DURATION']
    audio_duration_secs = len(waveform) / SC['AUDIO_SAMPLE_RATE']
    if audio_duration_secs > duration_secs:
        max_start = audio_duration_secs - duration_secs
        start_time_secs = np.random.uniform(0.0, max_start)
        start_index = int(start_time_secs * SC['AUDIO_SAMPLE_RATE'])
        end_index = start_index + int(duration_secs * SC['AUDIO_SAMPLE_RATE'])
        subsection = waveform[start_index: end_index]
    else:
        padding_length = int(duration_secs * SC['AUDIO_SAMPLE_RATE']) - len(waveform)
        padding = np.zeros(padding_length, dtype=waveform.dtype)
        subsection = np.concatenate([waveform, padding], axis=0)
    sample = subsection

    # Define audio augmentations using the audiomentations library
    augmentations = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
    ])
    sample = augmentations(samples=sample, sample_rate=SC['AUDIO_SAMPLE_RATE'])

    # Convert audio waveform to Mel spectrogram
    image = librosa.feature.melspectrogram(
        y=sample,
        sr=SC['AUDIO_SAMPLE_RATE'],
        n_fft=SC['AUDIO_NFFT'],
        hop_length=SC['AUDIO_STRIDE'],
        n_mels=SC['AUDIO_MELS'],
        fmin=SC['AUDIO_FMIN'],
        fmax=SC['AUDIO_FMAX'],
        win_length=SC['AUDIO_WINDOW'])
    image = librosa.power_to_db(
        image,
        top_db=SC['AUDIO_TOP_DB'],
        ref=1.0)
    expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
    image = np.moveaxis(image, 1, 0)
    sample = image[0:expected_clip_samples, :]
    image = np.expand_dims(sample, -1)
    image = np.repeat(image, SC['MODEL_INPUT_IMAGE_CHANNELS'], axis=2)
    expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
    image = tf.ensure_shape(image, [expected_clip_samples, SC['AUDIO_MELS'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
    image = tf.image.resize(image, (SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_HEIGHT']),
                            method=tf.image.ResizeMethod.LANCZOS5)
    image = image - tf.reduce_min(image)
    sample = image / (np.max(image) + 0.0000001)

    return sample


# Event detection. Automatically record the audio if the threshold is reached
def record_and_process_audio(threshold=500):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    skip_next_n_frames = 0
    print("Listening for events...")
    while True:
        audio_data = stream.read(CHUNK)
        if skip_next_n_frames > 0:
            skip_next_n_frames -= 1
            frames.append(audio_data)
            if skip_next_n_frames == 0:
                print("Processing captured audio...")
                # Turn the frames into waveform
                waveform = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
                waveform = librosa.resample(waveform, orig_sr=RATE, target_sr=params.SAMPLE_RATE)
                # Classify the sound
                should_classify_further = recognize_audio(waveform,
                                                          ['Animal', 'Bird', 'Bird vocalization, bird call, bird song'])
                # If the sound belongs to animals, using specialized engine to classify
                if should_classify_further:
                    audio_image = process_audio(waveform)
                    audio_image = np.expand_dims(audio_image, axis=0)
                    predictions = model.predict(audio_image)
                    predicted_index = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_index]
                    print(f"The predicted class is: {predicted_class}")
                else:
                    print("The audio doesn't match the criteria for further classification.")

                frames = []
            continue
        data_int = np.frombuffer(audio_data, dtype=np.int16)
        if max(data_int) > threshold:
            print("Event detected!")
            skip_next_n_frames = int((RATE / CHUNK) * 10)
            frames.append(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    record_and_process_audio()