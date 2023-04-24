
##################################################################################################
# This is the main Echo Engine application
#
# This program assumes you have already trained a model using the generic_engine_pipeline notebook
# Author: akudilczak
##################################################################################################


##################################################################################################
# library imports
##################################################################################################

# disable warnings
import warnings

warnings.filterwarnings("ignore")

import requests
import base64
import io
import json
import os
from platform import python_version

import diskcache as dc
# image processing related libraries
import librosa
import numpy as np
# generic libraries
import paho.mqtt.client as paho
# tensor flow / keras related libraries
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_io as tfio
from google.cloud import storage
from keras.utils import dataset_utils

# print system information
print('Python Version           : ', python_version())
print('TensorFlow Version       : ', tf.__version__)
print('TensorFlow IO Version    : ', tfio.__version__)
print('Librosa Version          : ', librosa.__version__)


##################################################################################################
# system constants copied from generic pipeline
##################################################################################################

# Load the JSON file into a dictionary
config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_engine.json')
with open(config_file_path, 'r') as f:
    SC = json.load(f)
    

class EchoEngine():

    ##################################################################################################
    ##################################################################################################
    def __init__(self) -> None:  
        pass
    
    ##################################################################################################
    # This function uses the google bucket with audio files and
    # leverages the folder names as the official species names
    # Note: to run this you will need to first authenticate
    # See https://github.com/DataBytes-Organisation/Project-Echo/tree/main/src/Prototypes/data#readme
    ##################################################################################################
    def gcp_load_species_list(self):

        species_names = set()

        bucket_name = SC['BUCKET_NAME']
        os.environ["GCLOUD_PROJECT"] = SC['GCLOUD_PROJECT']

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()  # Get list of files
        for blob in blobs:
            folder_name = blob.name.split('/')[0]
            species_names.add(folder_name)
        
        result = list(species_names)
        result.sort()
        
        return result
    

    ########################################################################################
    # Function to predict class and probability given a prediction
    ########################################################################################
    def predict_class(self, predictions):
        # Get the index of the class with the highest predicted probability
        predicted_index = int(tf.argmax(tf.squeeze(predictions)).numpy())
        # Get the class name using the predicted index
        predicted_class = self.class_names[predicted_index]
        # Calculate the predicted probability for the selected class
        predicted_probability = 100.0 * tf.nn.softmax(predictions)[predicted_index].numpy()
        # Round the probability to 2 decimal places
        predicted_probability = str(round(predicted_probability, 2))
        return predicted_class, predicted_probability


    ########################################################################################
    ########################################################################################
    def build_model(self):
        # Build a classification model using a pre-trained EfficientNetV2
        model = tf.keras.Sequential(
            [
                # Input layer with specified image dimensions
                tf.keras.layers.InputLayer(input_shape=(SC['MODEL_INPUT_IMAGE_HEIGHT'], 
                                                        SC['MODEL_INPUT_IMAGE_WIDTH'], 
                                                        SC['MODEL_INPUT_IMAGE_CHANNELS'])),

                # Use the EfficientNetV2 model as a feature generator (needs 260x260x3 images)
                hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2", False),

                # Add the classification layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),

                # Fully connected layer with multiple of the number of classes
                tf.keras.layers.Dense(len(self.class_names) * 8,
                                    activation="relu"),
                tf.keras.layers.BatchNormalization(),

                # Another fully connected layer with multiple of the number of classes
                tf.keras.layers.Dense(len(self.class_names) * 4,
                                    activation="relu"),
                tf.keras.layers.BatchNormalization(),

                # Add dropout to reduce overfitting
                tf.keras.layers.Dropout(0.50),

                # Output layer with one node per class, without activation
                tf.keras.layers.Dense(len(self.class_names), activation=None),
            ]
        )
        # Set the input shape for the model
        model.build([None, 
                    SC['MODEL_INPUT_IMAGE_HEIGHT'],
                    SC['MODEL_INPUT_IMAGE_WIDTH'], 
                    SC['MODEL_INPUT_IMAGE_CHANNELS']])

        return model


    ########################################################################################
    ########################################################################################
    def build_model_with_weights(self):

        # Build a new model for inference
        full_model = self.build_model()
        
        # Load the saved weights into the model
        full_model.load_weights(SC["MODEL_WEIGHTS"])
        
        return full_model


    ########################################################################################
    # this method takes in binary audio data and encodes to string
    ########################################################################################
    def string_to_audio(self, audio_string) -> bytes:
        base64_img_bytes = audio_string.encode('utf-8')
        decoded_data = base64.decodebytes(base64_img_bytes)
        return decoded_data
    
    
    ########################################################################################
    # this function is adapted from generic_engine_pipeline.ipynb
    # TODO: need to create a pipeline library and link same code into engine
    ########################################################################################
    def try_decode_audio(file_contents, decode_func):
        try:
            return decode_func(input=file_contents)
        except Exception as e:
            # print(f"An error occurred while decoding the audio file using {decode_func.__name__}:")
            # print(str(e))
            return None
    
    
    ########################################################################################
    # this function is adapted from generic_engine_pipeline.ipynb
    # TODO: need to create a pipeline library and link same code into engine
    ########################################################################################
    def load_random_subsection(self, tmp_audio_t, duration_secs):
    
        # Determine the audio file's duration in seconds
        audio_duration_secs = tf.shape(tmp_audio_t)[0] / SC['AUDIO_SAMPLE_RATE']
        
        if audio_duration_secs>duration_secs:
        
            # Calculate the starting point of the 5-second subsection
            max_start = tf.cast(audio_duration_secs - duration_secs, tf.float32)
            start_time_secs = tf.random.uniform((), 0.0, max_start, dtype=tf.float32)
            
            start_index = tf.cast(start_time_secs * SC['AUDIO_SAMPLE_RATE'], dtype=tf.int32)
    
            # Load the 5-second subsection
            end_index = tf.cast(start_index + tf.cast(duration_secs, tf.int32) * SC['AUDIO_SAMPLE_RATE'], tf.int32)
            
            subsection = tmp_audio_t[start_index : end_index]
        
        else:
            # Pad the subsection with silence if it's shorter than 5 seconds
            padding_length = duration_secs * SC['AUDIO_SAMPLE_RATE'] - tf.shape(tmp_audio_t)[0]
            padding = tf.zeros([padding_length], dtype=tmp_audio_t.dtype)
            subsection = tf.concat([tmp_audio_t, padding], axis=0).numpy()

        return subsection
    
    
    ########################################################################################
    # this function is adapted from generic_engine_pipeline.ipynb
    # TODO: need to create a pipeline library and link same code into engine
    ########################################################################################
    def combined_pipeline(self, audio_file, audio_clip):

        # Use BytesIO to create a file-like object from the binary data
        file_like_audio_data = io.BytesIO(audio_clip)

        # Load the audio data with librosa
        tmp_audio_t, sample_rate = librosa.load(file_like_audio_data, sr=None)
        
        # resample the sample rate
        tmp_audio_t = tfio.audio.resample(tmp_audio_t, sample_rate, SC['AUDIO_SAMPLE_RATE'])
        
        # if we didn't get exactly 5 seconds worth, this will choose a random subsection to process
        # ultimately, this ensures the image sizes are fixed when passed to the classifier
        tmp_audio_t = self.load_random_subsection(tmp_audio_t, SC['AUDIO_CLIP_DURATION'])
    
        # Convert to spectrogram
        image = tfio.audio.spectrogram(
            tmp_audio_t,
            nfft=SC['AUDIO_NFFT'], 
            window=SC['AUDIO_WINDOW'], 
            stride=SC['AUDIO_STRIDE'])

        # Convert to melspectrogram
        image = tfio.audio.melscale(
            image, 
            rate=SC['AUDIO_SAMPLE_RATE'], 
            mels=SC['AUDIO_MELS'], 
            fmin=SC['AUDIO_FMIN'], 
            fmax=SC['AUDIO_FMAX'])
        
        # reshape into standard 3 channels to add the color channel
        image = tf.expand_dims(image, -1)
        
        # most pre-trained model classifer model expects 3 color channels
        image = tf.repeat(image, SC['MODEL_INPUT_IMAGE_CHANNELS'], axis=2)
        
        # calculate the image shape and ensure it is correct
        expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
        image = tf.ensure_shape(image, [expected_clip_samples, SC['AUDIO_MELS'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
        
        # note here a high quality LANCZOS5 is applied to resize the image to match model image input size
        image = tf.image.resize(image, (SC['MODEL_INPUT_IMAGE_WIDTH'],SC['MODEL_INPUT_IMAGE_HEIGHT']), 
                                method=tf.image.ResizeMethod.LANCZOS5)

        # rescale to range [0,1]
        image = image - tf.reduce_min(image) 
        image = image / (tf.reduce_max(image)+0.0000001)
        
        return image
    

    ########################################################################################
    ########################################################################################
    def on_subscribe(self, client, userdata, mid, granted_qos):
        print(f"Subscribed: message id {mid} with qos {granted_qos}")


    ########################################################################################
    ########################################################################################
    def on_message(self, client, userdata, msg):
        print("Recieved audio message, processing via engine model...")    
        json_object = json.loads(msg.payload)
        print(json_object['timestamp'])
        
        # convert to string representation of audio to binary for processing
        audio_clip = self.string_to_audio(json_object['audioClip'])
        
        audio_file = json_object['audioFile']
        
        image = self.combined_pipeline(audio_file, audio_clip)
        
        image = tf.expand_dims(image, 0) 
        
        image_list = image.numpy().tolist()
        
        data = json.dumps({"signature_name": "serving_default", "inputs": image_list})

        url = SC['MODEL_SERVER']
        headers = {"content-type": "application/json"}
        json_response = requests.post(url, data=data, headers=headers)
    
        predictions = json.loads(json_response.text)['outputs'][0]
                
        # Predict class and probability using the prediction function
        predicted_class, predicted_probability = self.predict_class(predictions)

        print(f'Predicted class : {predicted_class}')
        print(f'Predicted probability : {predicted_probability}')
        

    ########################################################################################
    ########################################################################################
    def execute(self):
        print("Engine started.")
        client = paho.Client()
        client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message
        client.connect(SC['MQTT_CLIENT_URL'], SC['MQTT_CLIENT_PORT'])
        
        print(f'Subscribing to MQTT: {SC["MQTT_CLIENT_URL"]} {SC["MQTT_PUBLISH_URL"]}')
        client.subscribe(SC['MQTT_PUBLISH_URL'], qos=1)
        
        print("Retrieving species names from GCP")
        self.class_names = self.gcp_load_species_list()
        
        for cs in self.class_names:
            print(f" class name {cs}")

        #print("Building classifer model")
        #self.model = self.build_model_with_weights()
        # Display the model summary
        #self.model.summary()

        print("Engine waiting for audio to arrive...")
        client.loop_forever()


if __name__ == "__main__":
    
    engine = EchoEngine()
    engine.execute()
