import keras
import tensorflow as tf
import ffmpeg
import tensorflow_io as tfio
from keras.models import load_model as lm
import tfimm
from pydub import AudioSegment, effects
import os

import warnings
warnings.filterwarnings("ignore")

PATH_TO_MODEL = os.path.join(os.getcwd(), 'project-echo-package', 'project_echo', 'Models', 'baseline_timm_model_dataset_2_15_classes.hdf5')

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
# CLASSIFIER MODEL - leveraging EfficientNetV2
########################################################################################
class EchoTfimmModel(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):  
        super(EchoTfimmModel, self).__init__(*args, **kwargs)
        
        self.fm = tfimm.create_model("efficientnet_v2_s_in21k", pretrained=True, in_channels=MODEL_INPUT_IMAGE_CHANNELS)
        self.flat = tf.keras.layers.Flatten() 
        self.classifier = EchoClassifierLayer()

    def call(self, inputs, training=False):  
        x = self.fm.forward_features(inputs) 
        x = self.flat(x)
        x = self.classifier(x)               
        return x

def load_model():
    test_model = EchoTfimmModel()
    test_model.build([None, 224, 224, 1])
    test_model.load_weights(PATH_TO_MODEL)
    return test_model

def process_raw_audio(_model_, path_to_audio_file, sr: int = 16000):
    NFFT = 512
    WINDOW = 512
    STRIDE = 512
    SAMPLE_RATE = int(44100/2)
    MELS = 128
    FMIN = 0
    FMAX = int(SAMPLE_RATE)/2
    TOP_DB = 80
    CLIP_LENGTH = 5000
    BITRATE = '32k'

    _ret_data_ = []

    if not os.path.exists(path_to_audio_file): raise ValueError('Audio file does not exist')

    def dataset_transforms(image, _model_):  
        # reshape into standard 3 channels
        image = tf.expand_dims(image, -1)
        
        image = tf.ensure_shape(image, [216, 128, 1])
        image = tf.image.resize(image, 
                                (MODEL_INPUT_IMAGE_HEIGHT, 
                                MODEL_INPUT_IMAGE_WIDTH), 
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # rescale to range [0,1]
        image = image - tf.reduce_min(image) 
        image = image / (tf.reduce_max(image)+tf.keras.backend.epsilon()) 
        
        return image

    raw_sound = AudioSegment.from_file(path_to_audio_file, format=path_to_audio_file.split('.')[-1])
    raw_sound = effects.normalize(raw_sound)

    arr_split_file = [raw_sound[idx:idx + CLIP_LENGTH] for idx in range(0, len(raw_sound), CLIP_LENGTH)]             
    for count_sample, sample in enumerate(arr_split_file):
        # padding audio < 5s
        if len(sample) < CLIP_LENGTH:
            silence = AudioSegment.silent(duration=((CLIP_LENGTH-len(sample))))
            sample = sample + silence  # Adding silence after the audio

        sample.export(os.path.join(os.getcwd(), 'o.flac'), format='flac', bitrate=BITRATE, parameters = [])

        _tmp_path_ = os.path.join(os.getcwd(), 'o.flac')
        file_contents=tf.io.read_file(_tmp_path_)
        try:
            tmp_audio_t = tfio.audio.decode_flac(input=file_contents, dtype=tf.int16)
        except:
            tmp_audio_t = tfio.audio.decode_flac(input=file_contents, dtype=tf.int32)
            
        tmp_audio_t = tf.cast(tmp_audio_t, tf.float32)
            
        tmp_audio_t = tfio.audio.resample(tmp_audio_t, tfio.audio.AudioIOTensor(_tmp_path_)._rate.numpy(), SAMPLE_RATE)
        os.remove(_tmp_path_)


        # Convert to spectrogram
        spectrogram = tfio.audio.spectrogram(
            tmp_audio_t[:, 0], nfft=NFFT, window=WINDOW, stride=STRIDE)

        mel_spectrogram = tfio.audio.melscale(
                        spectrogram, rate=SAMPLE_RATE, mels=MELS, fmin=FMIN, fmax=FMAX)
        
        _tmp_path_ = os.path.join(os.getcwd(), 'o.pt')
        tf.io.write_file(_tmp_path_, tf.io.serialize_tensor(mel_spectrogram))

        tmp_data = tf.io.parse_tensor(tf.io.read_file(_tmp_path_), tf.float32)

        _mod_data_ = dataset_transforms(tmp_data, _model_)

        _ret_data_.append(_mod_data_)
        os.remove(_tmp_path_)
        
    return _ret_data_


def predict(_model_, path_to_file):
    _predict_data_ = process_raw_audio(_model_, path_to_file)

    for x in _predict_data_:
        print(_model_.predict(x))
        input()
    


path_to_raw_audio_file = '/Users/stephankokkas/Downloads/Alauda_arvensis_Bru_DAT0042_04_short.mp3'

my_model = load_model()
classification = predict(my_model, path_to_raw_audio_file)


