# tensorflow support
import tensorflow as tf
import tensorflow_io as tfio

import path
import sys
 
# directory reach
directory = path.Path(__file__).abspath()
 
# setting path
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)

# import prototype model
from prototype_models import echo_tfimm_model

class EchoAudioPipeline():
   
   def __init__(self):
        self.NFFT = 512
        self.WINDOW = 512
        self.STRIDE = 512
        self.SAMPLE_RATE = int(44100/2)
        self.MELS = 128
        self.FMIN = 0
        self.FMAX = int(self.SAMPLE_RATE)/2
        self.TOP_DB = 80
   
   def convert_audio_to_melspec(self, filename):
       
        # read the file data
        file_contents=tf.io.read_file(filename)
        
        # this prototype assumes that audio is exactly 5 seconds and in flac format
        try:
            tmp_audio_t = tfio.audio.decode_flac(input=file_contents, dtype=tf.int16)
        except:
            tmp_audio_t = tfio.audio.decode_flac(input=file_contents, dtype=tf.int32)

        tmp_audio_t = tf.cast(tmp_audio_t, tf.float32)

        # resample the sample rate
        tmp_audio_t = tfio.audio.resample(tmp_audio_t, tfio.audio.AudioIOTensor(filename)._rate.numpy(), self.SAMPLE_RATE)

        # Convert to spectrogram
        spectrogram = tfio.audio.spectrogram(
            tmp_audio_t[:, 0], nfft=self.NFFT, window=self.WINDOW, stride=self.STRIDE)
        
        # Convert to mel-spectrogram
        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=self.SAMPLE_RATE, mels=self.MELS, fmin=self.FMIN, fmax=self.FMAX)
                
        # reshape to single greyscale 1 channels
        mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)

        # resize to match model input size
        mel_spectrogram = tf.ensure_shape(mel_spectrogram, [216, 128, 1])
        mel_spectrogram = tf.image.resize(mel_spectrogram, 
                                (echo_tfimm_model.MODEL_INPUT_IMAGE_HEIGHT, 
                                echo_tfimm_model.MODEL_INPUT_IMAGE_WIDTH), 
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # rescale to range [0,1]
        mel_spectrogram = mel_spectrogram - tf.reduce_min(mel_spectrogram) 
        mel_spectrogram = mel_spectrogram / (tf.reduce_max(mel_spectrogram)+tf.keras.backend.epsilon()) 
        
        # Add single dimension for 'batch'
        mel_spectrogram = tf.expand_dims(mel_spectrogram, 0)
        
        return mel_spectrogram
   
