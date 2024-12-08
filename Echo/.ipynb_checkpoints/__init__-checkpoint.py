import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import ffmpeg
from keras.models import load_model as lm
from pydub import AudioSegment, effects
import tensorflow as tf
import tensorflow_io as tfio
from os.path import isfile, join
from os import listdir

target_classes = ['nightjar', 'skylark', 'yellow-faced honeyeater', 'feral goat',
                  'sambar deer', 'grey shrikethrush', 'australian raven', 'fallow deer',
                  'yellow robin', 'cat', 'whistler', 'white-plumed honeyeater',
                  'brown rat', 'pied currawong', 'wild pig']

########################################################################################
# MODEL PARAMETERS
########################################################################################
MODEL_INPUT_IMAGE_WIDTH = 224
MODEL_INPUT_IMAGE_HEIGHT = 224
MODEL_INPUT_IMAGE_CHANNELS = 1

########################################################################################
# CLASSIFIER LAYER
########################################################################################
class EchoClassifierLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EchoClassifierLayer, self).__init__()
        
        dropout=0.5
        
        self.fc1 = tf.keras.layers.Dense(128, 
                                         kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                         activation=tf.keras.activations.relu)

        self.fc2 = tf.keras.layers.Dense(128, 
                                         kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                         activation=tf.keras.activations.relu)
        
        self.do2 = tf.keras.layers.Dropout(dropout)        
        
        self.out = tf.keras.layers.Dense(15, 
                                         activation=tf.keras.activations.linear)

    def call(self, inputs):
        x = self.fc1(inputs)               
        x = self.fc2(x)               
        x = self.do2(x)           
        x = self.out(x)
        return x

########################################################################################
# MODEL
########################################################################################
class EchoModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):  
        super(EchoModel, self).__init__(*args, **kwargs)
        
        # EfficientNetV2
        self.fm = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=(MODEL_INPUT_IMAGE_HEIGHT, MODEL_INPUT_IMAGE_WIDTH, MODEL_INPUT_IMAGE_CHANNELS)
        )
        self.flat = tf.keras.layers.Flatten() 
        self.classifier = EchoClassifierLayer()

    def call(self, inputs, training=False):  
        x = self.fm(inputs)
        x = self.flat(x)
        x = self.classifier(x)               
        return x

def load_model():
    try:
        # Set paths relative to the package installation directory.
        base_path = os.path.dirname(os.path.abspath(__file__))  # use __file__
        model_path = os.path.join(base_path, 'Models', 'echo_model', '1')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
            
        # Load a model in SavedModel format.
        model = tf.keras.models.load_model(model_path)
        print("Model loading successful.")
        return model
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise

def process_raw_audio(_model_, path_to_audio_file, sr: int = 16000):
    import librosa
    import numpy as np
    
    NFFT = 512
    WINDOW = 512
    STRIDE = 512
    SAMPLE_RATE = int(44100/2)
    MELS = 128
    FMIN = 0
    FMAX = int(SAMPLE_RATE)/2
    CLIP_LENGTH = 5000
    _ret_data_ = []

    if not os.path.exists(path_to_audio_file):
        raise ValueError('Audio file does not exist')

    def dataset_transforms(image, _model_):
        # Convert to tensor
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, -1)
        # Resize to 260x260
        image = tf.image.resize(image, (260, 260))
        # Convert to RGB
        image = tf.image.grayscale_to_rgb(image)
        # Normalize
        image = image - tf.reduce_min(image)
        image = image / (tf.reduce_max(image) + tf.keras.backend.epsilon())
        return image

    # Load audio file
    raw_sound = AudioSegment.from_file(path_to_audio_file, 
                                     format=path_to_audio_file.split('.')[-1])
    raw_sound = effects.normalize(raw_sound)
    
    # Split into 5-second segments
    arr_split_file = [raw_sound[idx:idx + CLIP_LENGTH] 
                     for idx in range(0, len(raw_sound), CLIP_LENGTH)]

    for count_sample, sample in enumerate(arr_split_file):
        # Add padding if needed
        if len(sample) < CLIP_LENGTH:
            silence = AudioSegment.silent(duration=(CLIP_LENGTH-len(sample)))
            sample = sample + silence

        # Export to temporary WAV file
        tmp_wav = os.path.join(os.getcwd(), 'temp.wav')
        sample.export(tmp_wav, format='wav')
        
        try:
            # Load audio using librosa
            y, sr = librosa.load(tmp_wav, sr=SAMPLE_RATE)
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_fft=NFFT,
                hop_length=STRIDE,
                win_length=WINDOW,
                n_mels=MELS,
                fmin=FMIN,
                fmax=FMAX
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Process for model
            _mod_data_ = dataset_transforms(mel_spec, _model_)
            _mod_data_ = tf.expand_dims(_mod_data_, 0)
            
            _ret_data_.append(_mod_data_)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
    
    return _ret_data_


def predict(_model_, path_to_file, traverse_path:bool = False):
    def translate_results(result):
        # Binary classification with a threshold of 0.5
        value = result[0, 0]  # Get the single value
        score = float(value)  # numpy float to Python float
        
        # Classify using 0.5 as the threshold
        if score >= 0.5:
            target_class = target_classes[0]  # positive
        else:
            target_class = target_classes[1]  # negative
            
        # Display confidence as a percentage
        confidence = abs(score - 0.5) * 200  # Convert distance from 0.5 to confidence
        return target_class, str(round(confidence, 2))

    if not traverse_path:
        _predict_data_ = process_raw_audio(_model_, path_to_file)
        print(f'Your audio file is: {os.path.split(path_to_file)[-1]}')
        print(f'Your file is split into {len(_predict_data_)} windows of 5 seconds width per window. For each sliding window, we found:')
        for x in _predict_data_:
            prediction = _model_(x)
            _ret = translate_results(prediction)
            print(f'    A {_ret[0]} with a confidence of {_ret[1]}%')
    else:
        for _file_ in [f for f in listdir(path_to_file) if isfile(join(path_to_file, f))]:
            _predict_data_ = process_raw_audio(_model_, os.path.join(path_to_file, _file_))
            print(f'Your audio file is: {os.path.split(os.path.join(path_to_file, _file_))[-1]}')
            print(f'Your file is split into {len(_predict_data_)} windows of 5 seconds width per window. For each sliding window, we found:')
            for x in _predict_data_:
                prediction = _model_(x)
                _ret = translate_results(prediction)
                print(f'    A {_ret[0]} with a confidence of {_ret[1]}%')