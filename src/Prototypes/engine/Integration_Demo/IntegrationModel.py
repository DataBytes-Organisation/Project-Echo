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
from sklearn.preprocessing import LabelEncoder

# Set constants for audio processing
SC = {
    'AUDIO_CLIP_DURATION': 2,  # seconds
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
model = tf.keras.models.load_model('C:\\Users\\22396\\Documents\\GitHub\\Project-Echo2024\\Project-Echo\\src\\Components\\Engine\\models\\echo_model\\1') # Replace this with your own model path.
df = pd.read_csv('Classes.csv')
class_names = df["Class Name"].tolist()

#Load weather engine
weather_model = tf.keras.models.load_model('WeatherAudioDetectionModel.h5')

# Using YamNet to recognize audio for further classification
def recognize_audio(waveform, expected_class_names, top_n=10):
    prediction = np.mean(yamnet_model.predict(np.reshape(waveform, [1, -1]), steps=1)[0], axis=0)
    sorted_indices = np.argsort(prediction)[-top_n:][::-1]
    matching_classes = {yamnet_classes[i]: prediction[i] for i in sorted_indices if yamnet_classes[i] in expected_class_names}

    # Check for bird and weather-related sounds
    bird_related = {'Animal', 'Bird', 'Bird vocalization, bird call, bird song'}
    is_bird_call = any(cls in bird_related for cls in matching_classes)

    return is_bird_call



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

# Detect events from an audio. Set a flag to mark the start and end of an event.
def enhanced_event_detection(waveform, sample_rate, threshold=0.5, min_event_duration=1.0, max_silence_duration=0.5):
    num_samples = len(waveform)
    events = []
    is_event = False
    event_start = 0

    silence_samples = int(max_silence_duration * sample_rate)
    min_event_samples = int(min_event_duration * sample_rate)
    silence_counter = 0

    for i in range(num_samples):
        if abs(waveform[i]) > threshold:
            if not is_event:
                is_event = True
                event_start = i
            silence_counter = 0
        else:
            if is_event:
                silence_counter += 1
                if silence_counter > silence_samples:
                    if i - event_start > min_event_samples:
                        events.append((event_start, i - silence_counter))
                    is_event = False
                    silence_counter = 0

    # Handle case where the last detected event reaches the end of the waveform
    if is_event:
        events.append((event_start, num_samples))

    print(f"Detected {len(events)} events.")

    return events
def pad_audio(audio, duration=2, sr=16000):
    required_samples = sr * duration
    if len(audio) < required_samples:
        audio = np.pad(audio, (0, required_samples - len(audio)), 'constant')
    else:
        audio = audio[:required_samples]
    return audio

def prepare_audio_segment(audio_segment):
    audio_padded = pad_audio(audio_segment, duration=SC['AUDIO_CLIP_DURATION'], sr=SC['AUDIO_SAMPLE_RATE'])
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_padded, sr=SC['AUDIO_SAMPLE_RATE'],
        n_fft=SC['AUDIO_NFFT'],
        hop_length=SC['AUDIO_STRIDE'],
        n_mels=SC['AUDIO_MELS'],
        fmin=SC['AUDIO_FMIN'],
        fmax=SC['AUDIO_FMAX']
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, top_db=SC['AUDIO_TOP_DB'])
    spectrogram_resized = tf.image.resize(log_mel_spectrogram[np.newaxis, :, :, np.newaxis], [SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_WIDTH']])
    spectrogram_resized = np.repeat(spectrogram_resized, SC['MODEL_INPUT_IMAGE_CHANNELS'], axis=-1)
    return spectrogram_resized

def process_audio_file(audio_filepath, threshold=0.05, weather_threshold=0.5):
    print("Processing audio file:", audio_filepath)
    # Load the audio file
    waveform, _ = librosa.load(audio_filepath, sr=SC['AUDIO_SAMPLE_RATE'])
    print(f"Loaded waveform with {len(waveform)} samples.")

    # Detect events in the audio
    events = enhanced_event_detection(waveform, SC['AUDIO_SAMPLE_RATE'], threshold)

    for start_sample, end_sample in events:
        segment = waveform[start_sample:end_sample]

        audio_image = prepare_audio_segment(segment)

        weather_predictions = weather_model.predict(audio_image)
        handle_weather_predictions(weather_predictions, weather_threshold)

        audio_image = process_audio(segment)

        is_bird_call = recognize_audio(segment,['Animal', 'Bird', 'Bird vocalization, bird call, bird song'])
        print(f"YAMNet prediction for segment: Bird call: {is_bird_call}")

        if is_bird_call:
            # Prepare input for bird call model
            bird_audio_image = np.expand_dims(audio_image, axis=0)
            predictions = model.predict(bird_audio_image)
            if predictions.size > 0 and len(predictions[0]) > 0:
                predicted_index = np.argmax(predictions[0])
                if predicted_index < len(class_names):
                    predicted_class = class_names[predicted_index]
                    start_time = round(start_sample / SC['AUDIO_SAMPLE_RATE'], 1)
                    end_time = round(end_sample / SC['AUDIO_SAMPLE_RATE'], 1)
                    print(f"Specialized model prediction between {start_time}s to {end_time}s: {predicted_class}")
                else:
                    print("Predicted index is out of range for class names.")
            else:
                print("No valid predictions were returned.")


def handle_weather_predictions(weather_predictions, weather_threshold=0.5):
    encoder = LabelEncoder()
    encoder.fit(['Earthquake', 'Rain', 'Thunder', 'Tornado'])
    predicted_class = np.argmax(weather_predictions, axis=1)
    if predicted_class.size == 1:
        predicted_class = predicted_class.item()
    predicted_label = encoder.inverse_transform([predicted_class])
    weather_probability = max(weather_predictions.flatten())
    if weather_probability < weather_threshold:
        print("No particular weather events detected with high confidence.")
    else:
        print(f"Current weather events: {predicted_label[0]}")


if __name__ == "__main__":
    audio_filepath = r"C:\Users\22396\Downloads\audio sample.wav"
    try:
        process_audio_file(audio_filepath)
    except Exception as e:
        print("An error occurred:", e)
