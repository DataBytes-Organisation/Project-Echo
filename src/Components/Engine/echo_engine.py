
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

# os environment
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import time
import requests
import base64
import io
import json
import base64
import tempfile
import pickle
import math
import numpy as np
import pandas as pd
import soundfile as sf
import sys
sys.path.append("yamnet/")

from platform import python_version

import diskcache as dc
# image processing related libraries
import librosa
# generic libraries
import paho.mqtt.client as paho
# tensor flow / keras related libraries
import tensorflow as tf
from tensorflow.keras.models import load_model

from google.cloud import storage

# yamnet related imports
from yamnet_dir import params as params
from yamnet_dir import yamnet as yamnet_model

# lat long approximation
import random
from geopy.distance import geodesic

# database libraries
import pymongo
from helpers import melspectrogram_to_cam

# for weather label
from sklearn.preprocessing import LabelEncoder


# print system information
print('Python Version           : ', python_version())
print('TensorFlow Version       : ', tf.__version__)
print('Librosa Version          : ', librosa.__version__)

# Load the necessary data and models
with open('yamnet_dir/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

with open('yamnet_dir/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet_dir/yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_dir/yamnet_class_map.csv')
model = load_model('yamnet_dir/model_3_82_16000.h5')

# Load the YAMNet model
# yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
# yamnet_model = hub.load(yamnet_model_handle)
#TODO: Fix for above macOS, as installing tensorflow hub causes issue
yamnet_model =tf.saved_model.load('yamnet_dir/model')


class EchoEngine():

    ##################################################################################################
    ##################################################################################################
    def __init__(self) -> None:  
        
        # Load the engine config JSON file into a dictionary
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_engine.json')
            with open(file_path, 'r') as f:
                self.config = json.load(f)
            self.config['AUDIO_WINDOW'] = None 
            print(f"Echo Engine configuration successfully loaded", flush=True)
        except:
            print(f"Could not engine config : {file_path}") 
        
        # Load the project echo credentials into a dictionary
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_credentials.json')
            with open(file_path, 'r') as f:
                self.credentials = json.load(f)
            print(f"Echo Engine credentials successfully loaded", flush=True)
        except:
            print(f"Could not engine credentials : {file_path}") 
         
        # Setup database client and connect
        try:
            # database connection string
            self.connection_string=f"mongodb://{self.credentials['DB_USERNAME']}:{self.credentials['DB_PASSWORD']}@{self.config['DB_HOSTNAME']}/EchoNet"

            myclient = pymongo.MongoClient(self.connection_string)
            self.echo_store = myclient["EchoNet"]
            print(f"Found echo store database names: {myclient.list_database_names()}", flush=True)
        except:
            print(f"Failed to establish database connection", flush=True)

    
    ##################################################################################################
    # This function uses the google bucket with audio files and
    # leverages the folder names as the official species names
    # Note: to run this you will need to first authenticate
    # See https://github.com/DataBytes-Organisation/Project-Echo/tree/main/src/Prototypes/data#readme
    ##################################################################################################
    def gcp_load_species_list(self):

        species_names = set()

        bucket_name = self.config['BUCKET_NAME']
        os.environ["GCLOUD_PROJECT"] = self.config['GCLOUD_PROJECT']

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
        predicted_probability = round(predicted_probability, 2)
        return predicted_class, predicted_probability


    ########################################################################################
    # this method takes in string and ecodes to audio binary data
    ########################################################################################
    def string_to_audio(self, audio_string) -> bytes:
        base64_img_bytes = audio_string.encode('utf-8')
        decoded_data = base64.decodebytes(base64_img_bytes)
        return decoded_data
    
    ########################################################################################
    # this method takes in string and ecodes to audio binary data
    ########################################################################################
    def recorded_string_to_audio(self, audio_string) -> bytes:
        decoded_data = base64.b64decode(audio_string)
        return decoded_data
    
    ########################################################################################
    # this function is adapted from generic_engine_pipeline.ipynb
    # TODO: need to create a pipeline library and link same code into engine
    ########################################################################################
    def load_random_subsection(self, tmp_audio_t, duration_secs):
    
        # Determine the audio file's duration in seconds
        audio_duration_secs = tf.shape(tmp_audio_t)[0] / self.config['AUDIO_SAMPLE_RATE']
        
        if audio_duration_secs>duration_secs:
        
            # Calculate the starting point of the 5-second subsection
            max_start = tf.cast(audio_duration_secs - duration_secs, tf.float32)
            start_time_secs = tf.random.uniform((), 0.0, max_start, dtype=tf.float32)
            
            start_index = tf.cast(start_time_secs * self.config['AUDIO_SAMPLE_RATE'], dtype=tf.int32)
    
            # Load the 5-second subsection
            end_index = tf.cast(start_index + tf.cast(duration_secs, tf.int32) * self.config['AUDIO_SAMPLE_RATE'], tf.int32)
            
            subsection = tmp_audio_t[start_index : end_index]
        
        else:
            # Pad the subsection with silence if it's shorter than 5 seconds
            padding_length = duration_secs * self.config['AUDIO_SAMPLE_RATE'] - tf.shape(tmp_audio_t)[0]
            padding = tf.zeros([padding_length], dtype=tmp_audio_t.dtype)
            subsection = tf.concat([tmp_audio_t, padding], axis=0).numpy()

        return subsection
    
    def load_specific_subsection(self, tmp_audio_t, start_time_secs, end_time_secs, sample_rate):
        # Ensure start and end times are within the audio duration
        audio_duration_secs = tf.shape(tmp_audio_t)[0] / sample_rate
        if end_time_secs > audio_duration_secs:
            end_time_secs = audio_duration_secs

        # Convert start and end times to sample indices
        start_index = tf.cast(start_time_secs * sample_rate, dtype=tf.int32)
        end_index = tf.cast(end_time_secs * sample_rate, dtype=tf.int32)

        # Load the specified subsection
        subsection = tmp_audio_t[start_index:end_index]

        # If the subsection is shorter than expected, pad it with silence
        expected_length = int((end_time_secs - start_time_secs) * sample_rate)
        if subsection.shape[0] < expected_length:
            padding_length = expected_length - subsection.shape[0]
            padding = tf.zeros([padding_length], dtype=tmp_audio_t.dtype)
            subsection = tf.concat([subsection, padding], axis=0)

        return subsection
    
    ########################################################################################
    # this function is adapted from generic_engine_pipeline.ipynb
    # TODO: need to create a pipeline library and link same code into engine
    ########################################################################################
    def combined_pipeline(self, audio_clip, mode):
        
        sample_rate = 0

        if(str(mode) == "Recording_Mode"):
            print("recording mode 1")
            # Create a file-like object from the bytes.
            file = io.BytesIO(audio_clip)

            # Load the audio data with librosa
            audio_clip, sample_rate = librosa.load(file, sr=self.config['AUDIO_SAMPLE_RATE'])
        
        elif(str(mode) == "Recording_Mode_V2"):
            print("recording mode 2")
            # Create a file-like object from the bytes.
            file = io.BytesIO(audio_clip)

            # Load the audio data with librosa
            audio_clip, sample_rate = librosa.load(file, sr=self.config['AUDIO_SAMPLE_RATE'])
        
        else:
            print("animal mode")
            # Create a file-like object from the bytes.
            file = io.BytesIO(audio_clip)

            # Load the audio data with librosa
            audio_clip, sample_rate = librosa.load(file, sr=self.config['AUDIO_SAMPLE_RATE'])

        # keep right channel only
        if audio_clip.ndim == 2 and audio_clip.shape[0] == 2:
            audio_clip = audio_clip[1, :]
        
        # cast to float32 type
        audio_clip = audio_clip.astype(np.float32)
        
        # analyse a random 5 second subsection
        audio_clip = self.load_random_subsection(audio_clip, duration_secs=self.config['AUDIO_CLIP_DURATION'])

        # Compute the mel-spectrogram
        image = librosa.feature.melspectrogram(
            y=audio_clip, 
            sr=self.config['AUDIO_SAMPLE_RATE'], 
            n_fft=self.config['AUDIO_NFFT'], 
            hop_length=self.config['AUDIO_STRIDE'], 
            n_mels=self.config['AUDIO_MELS'],
            fmin=self.config['AUDIO_FMIN'],
            fmax=self.config['AUDIO_FMAX'],
            win_length=self.config['AUDIO_WINDOW'])

        # Optionally convert the mel-spectrogram to decibel scale
        image = librosa.power_to_db(
            image, 
            top_db=self.config['AUDIO_TOP_DB'], 
            ref=1.0)
        
        # Calculate the expected number of samples in a clip
        expected_clip_samples = int(self.config['AUDIO_CLIP_DURATION'] * self.config['AUDIO_SAMPLE_RATE'] / self.config['AUDIO_STRIDE'])
        
        # swap axis and clip to expected samples to avoid rounding errors
        image = np.moveaxis(image, 1, 0)
        image = image[0:expected_clip_samples,:]
        
        # reshape into standard 3 channels to add the color channel
        image = tf.expand_dims(image, -1)
        
        # most pre-trained model classifer model expects 3 color channels
        image = tf.repeat(image, self.config['MODEL_INPUT_IMAGE_CHANNELS'], axis=2)
        
        # calculate the image shape and ensure it is correct
        expected_clip_samples = int(self.config['AUDIO_CLIP_DURATION'] * self.config['AUDIO_SAMPLE_RATE'] / self.config['AUDIO_STRIDE'])
        image = tf.ensure_shape(image, [expected_clip_samples, self.config['AUDIO_MELS'], self.config['MODEL_INPUT_IMAGE_CHANNELS']])
        
        # note here a high quality LANCZOS5 is applied to resize the image to match model image input size
        image = tf.image.resize(image, (self.config['MODEL_INPUT_IMAGE_WIDTH'],self.config['MODEL_INPUT_IMAGE_HEIGHT']), 
                                method=tf.image.ResizeMethod.LANCZOS5)

        # rescale to range [0,1]
        image = image - tf.reduce_min(image) 
        image = image / (tf.reduce_max(image)+0.0000001)
        
        return image, audio_clip, sample_rate


    # this method takes in binary audio data and encodes to string
    def audio_to_string(self, audio_binary) -> str:
        base64_encoded_data = base64.b64encode(audio_binary)
        base64_message = base64_encoded_data.decode('utf-8')
        return base64_message    


    ########################################################################################
    ########################################################################################
    def on_subscribe(self, client, userdata, mid, granted_qos):
        print(f"Subscribed: message id {mid} with qos {granted_qos}")


    ########################################################################################
    ########################################################################################
    def on_message(self, client, userdata, msg):
        print("Recieved audio message, processing via engine model...") 
        try:   
            audio_event = json.loads(msg.payload)
            print(audio_event['timestamp'])

            audio_clip = ""
            image = None
            sample_rate = 0

            if(audio_event['audioFile'] == "Recording_Mode"): # classic model
                # convert to string representation of audio to binary for processing
                print("Recording_Mode")
                audio_clip = self.string_to_audio(audio_event['audioClip'])
            
                image, audio_clip, sample_rate = self.combined_pipeline(audio_clip, "Recording_Mode")
            
                # update the audio event with the re-sampled audio
                audio_event["audioClip"] = self.audio_to_string(audio_clip)

                image = tf.expand_dims(image, 0) 
            
                cam = melspectrogram_to_cam.convert(image)
                image_list = image.numpy().tolist()
            
                # Run the model via tensorflow serve
                data = json.dumps({"signature_name": "serving_default", "inputs": image_list})
                url = self.config['MODEL_SERVER']
                headers = {"content-type": "application/json"}
                json_response = requests.post(url, data=data, headers=headers)
                model_result   = json.loads(json_response.text)
                predictions = model_result['outputs'][0]
                    
                # Predict class and probability using the prediction function
                predicted_class, predicted_probability = self.predict_class(predictions)

                print(f'Predicted class : {predicted_class}')
                print(f'Predicted probability : {predicted_probability}')
            
                # populate the database with the result
                self.echo_api_send_detection_event(
                    audio_event,
                    sample_rate,
                    predicted_class,
                    predicted_probability,
                    cam)
                    
            
                image = tf.expand_dims(image, 0) 
            
                image_list = image.numpy().tolist()

            elif(audio_event['audioFile'] == "Recording_Mode_V2"):
                # convert to string representation of audio to binary for processing
                print("Recording_Mode_V2")
                sample_rate = 16000
                audio_clip = self.string_to_audio(audio_event['audioClip'])
                file = io.BytesIO(audio_clip)
                #wav = 'yamnet_dir/cat-goat-dingo.wav'
                data_frame, audio_clip = self.sound_event_detection(file, sample_rate)
                iteration_count = 0
            
                for index, row in data_frame.iterrows():
                    #start_time	end_time	echonet_label_1	echonet_confidence_1
                    start_time = float(row['start_time'])
                    end_time = float(row['end_time'])
                    predicted_class = row['echonet_label_1']

                    if predicted_class == "Sus_Scrofa":
                        predicted_class = "Sus Scrofa"
                    
                    predicted_probability = round(float(row['echonet_confidence_1']) * 100.0, 2)

                    print(f'Predicted class : {predicted_class}')
                    print(f'Predicted probability : {predicted_probability}')

                    audio_subsection = self.load_specific_subsection(audio_clip, start_time, end_time, sample_rate)

                    # update the audio event with the re-sampled audio
                    audio_event["audioClip"] = self.audio_to_string(audio_subsection)

                    new_lat = audio_event['animalEstLLA'][0]
                    new_lon = audio_event['animalEstLLA'][1]

                    if(iteration_count > 0):
                        lat = audio_event['animalEstLLA'][0]
                        lon = audio_event['animalEstLLA'][1]

                        new_lat, new_lon = self.generate_random_location(lat, lon, 50, 100)

                    new_lla = [new_lat, new_lon, 0.0]

                    audio_event['animalEstLLA'] = new_lla
                    audio_event['animalTrueLLA'] = new_lla

                    # populate the database with the result
                    self.echo_api_send_detection_event(
                        audio_event,
                        sample_rate,
                        predicted_class,
                        predicted_probability)
                    
                    iteration_count = iteration_count + 1

            else: # simulate animals mode
                # convert to string representation of audio to binary for processing
                audio_clip = self.string_to_audio(audio_event['audioClip'])
            
                image, audio_clip, sample_rate = self.combined_pipeline(audio_clip, "Animal_Mode")
            
                cam = melspectrogram_to_cam.convert(image)
                # update the audio event with the re-sampled audio
                audio_event["audioClip"] = self.audio_to_string(audio_clip)
                
                image = tf.expand_dims(image, 0) 
            
                image_list = image.numpy().tolist()
            
                # Run the model via tensorflow serve
                data = json.dumps({"signature_name": "serving_default", "inputs": image_list})
                url = self.config['MODEL_SERVER']
                headers = {"content-type": "application/json"}
                json_response = requests.post(url, data=data, headers=headers)
                model_result   = json.loads(json_response.text)
                predictions = model_result['outputs'][0]
                    
                # Predict class and probability using the prediction function
                predicted_class, predicted_probability = self.predict_class(predictions)

                print(f'Predicted class : {predicted_class}')
                print(f'Predicted probability : {predicted_probability}')
            
                # populate the database with the result
                self.echo_api_send_detection_event(
                    audio_event,
                    sample_rate,
                    predicted_class,
                    predicted_probability,
                    cam)
            
                image = tf.expand_dims(image, 0) 
            
                image_list = image.numpy().tolist()
            
        except Exception as e:
            # Catch the exception and print it to the console
            print(f"An error occurred: {e}", flush=True)


    ########################################################################################
    # this function populates the database with the prediction results
    ########################################################################################
    def echo_api_send_detection_event(self, audio_event, sample_rate, predicted_class, predicted_probability,cam=None):
        
        detection_event = {
            "timestamp": audio_event["timestamp"],
            "species": predicted_class,
            "confidence": predicted_probability, 
            "sensorId": audio_event["sensorId"],
            "microphoneLLA": audio_event["microphoneLLA"],
            "animalEstLLA": audio_event["animalEstLLA"], 
            "animalTrueLLA": audio_event["animalTrueLLA"], 
            "animalLLAUncertainty": audio_event["animalLLAUncertainty"],
            "audioClip": audio_event["audioClip"],
            "sampleRate": sample_rate,
            "cam" : cam       
        }
        
        url = 'http://ts-api-cont:9000/engine/event'
        x = requests.post(url, json = detection_event)
        print(x.text)
    
    def weather_pipeline(self, audio_clip):
        """
            Processes an audio clip to generate a resized log-mel spectrogram. To be used similar to combined_pipeline() function

            Args:
                audio_clip (bytes): The audio clip in bytes format.

            Returns:
                tuple: A tuple containing:
                    - spectrogram_resized (numpy.ndarray): The resized log-mel spectrogram with shape (260, 260, 3).
                    - audio (numpy.ndarray): The processed audio data.
                    - sample_rate (int): The sample rate used for the audio processing.

            The function performs the following steps:
            1. Converts the audio clip from bytes to a file-like object.
            2. Loads the audio data using librosa with a specified sample rate.
            3. Pads or truncates the audio data to ensure it has the required number of samples.
            4. Computes the mel spectrogram of the audio data.
            5. Converts the mel spectrogram to a log-mel spectrogram.
            6. Resizes the log-mel spectrogram to a fixed size of 260x260 pixels and repeats it across three channels.
            7. Returns the resized log-mel spectrogram, the processed audio data, and the sample rate.
        """

        file = io.BytesIO(audio_clip)
        # Load the audio data with librosa
        audio, sample_rate = librosa.load(file, sr=self.config['WEATHER_SAMPLE_RATE'])
        required_samples = self.config['WEATHER_SAMPLE_RATE'] * self.config['WEATHER_CLIP_DURATION']
        if len(audio) < required_samples:
            audio = np.pad(audio, (0, required_samples - len(audio)), 'constant')
        else:
            audio = audio[:required_samples]

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=self.config['WEATHER_SAMPLE_RATE'],
            n_fft=self.config['AUDIO_NFFT'],
            hop_length=self.config['AUDIO_STRIDE'],
            n_mels=self.config['AUDIO_MELS'],
            fmin=self.config['AUDIO_FMIN'],
            fmax=self.config['AUDIO_FMAX']
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, top_db=self.config['AUDIO_TOP_DB'])
        spectrogram_resized = tf.image.resize(log_mel_spectrogram[np.newaxis, :, :, np.newaxis], [260, 260])
        spectrogram_resized = np.repeat(spectrogram_resized, 3, axis=-1)
        return spectrogram_resized, audio, self.config['WEATHER_SAMPLE_RATE']    

    def load_audio_file(self, file_path):
        wav, sr = librosa.load(file_path, sr=16000)
        return np.array([wav])

    def extract_features(self, model, X):
        features = []
        for wav in X:
            scores, embeddings, spectrogram = model(wav)
            features.append(embeddings.numpy().mean(axis=0))
        return np.array(features)

    def predict_on_audio(self, binary_audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            with open(temp_audio_file.name, 'wb') as f:
                f.write(binary_audio)
            X_new = self.load_audio_file(temp_audio_file.name)
            X_new_features = self.extract_features(yamnet_model, X_new)

            predictions = model.predict(X_new_features)
            top_two_prob_indices = np.argsort(predictions[0])[-2:]
            top_two_prob_values = predictions[0][top_two_prob_indices]

            top_two_class_names = le.inverse_transform(top_two_prob_indices)
        
            return [(class_names[top_two_prob_indices[1-i]], top_two_prob_values[1-i]) for i in range(2)]

    def sound_event_detection(self, filepath, sample_rate):
        data, sr = librosa.load(filepath, sr=16000)
        frame_len = int(sr * 1)
        num_chunks = len(data) // frame_len
        chunks = [data[i*frame_len:(i+1)*frame_len] for i in range(num_chunks)]

        # Adding the last chunk which can be less than 1 second
        last_chunk = data[num_chunks*frame_len:]
        if len(last_chunk) > 0:
            chunks.append(last_chunk)

        animal_related_classes = [
            'Dog', 'Cat', 'Bird', 'Animal', 'Birdsong', 'Canidae', 'Feline', 'Livestock',
            'Rodents, Mice', 'Wild animals', 'Pets', 'Frogs', 'Insect', 'Snake', 
            'Domestic animals, pets', 'crow'
        ]

        df_rows = []
        buffer = []
        start_time = None
        for cnt, frame_data in enumerate(chunks):
            frame_data = np.reshape(frame_data, (-1,)) # Flatten the array to 1D
            frame_data = np.array([frame_data]) # Wrapping it back into a 2D array
            outputs = yamnet(frame_data)
            yamnet_prediction = np.mean(outputs[0], axis=0)
            top2_i = np.argsort(yamnet_prediction)[::-1][:2]
            threshold=0.2
            if any(yamnet_prediction[np.where(yamnet_classes == cls)[0][0]] >= threshold for cls in animal_related_classes if cls in yamnet_classes):
                if start_time is None:
                    start_time = cnt
                buffer.append(frame_data)
            else:
                if start_time is not None:
                    segment_data = np.concatenate(buffer, axis=1)[0]
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        sf.write(temp_audio_file.name, segment_data, sr)
                        with open(temp_audio_file.name, 'rb') as binary_file:
                            top2_predictions = self.predict_on_audio(binary_file.read())

                    df_row = {'start_time': start_time, 'end_time': cnt}
                
                    for i, pred in enumerate(top2_predictions[:2]):
                        df_row[f'echonet_label_{i+1}'] = pred[0] if pred[0] is not None else None
                        df_row[f'echonet_confidence_{i+1}'] = pred[1] if pred[1] is not None else None

                    df_rows.append(df_row)
                    buffer = []
                    start_time = None

        # Handling the case where the last chunk contains an animal-related sound
        if start_time is not None:
            segment_data = np.concatenate(buffer, axis=1)[0]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                sf.write(temp_audio_file.name, segment_data, sr)
                with open(temp_audio_file.name, 'rb') as binary_file:
                    top2_predictions = self.predict_on_audio(binary_file.read())

            df_row = {'start_time': start_time, 'end_time': len(chunks)}
        
            for i, pred in enumerate(top2_predictions[:2]):
                df_row[f'echonet_label_{i+1}'] = pred[0] if pred[0] is not None else None
                df_row[f'echonet_confidence_{i+1}'] = pred[1] if pred[1] is not None else None

            df_rows.append(df_row)

        df = pd.DataFrame(df_rows)

        # keep right channel only
        if data.ndim == 2 and data.shape[0] == 2:
            data = data[1, :]
        
        # cast to float32 type
        data = data.astype(np.float32)

        return df, data
    
    def generate_random_location(self, lat, lon, min_distance, max_distance):
        # Generate a random direction in radians (0 to 2*pi)
        random_direction = random.uniform(0, 2 * 3.14159265359)

        # Generate a random distance between min and max distances
        random_distance = random.uniform(min_distance, max_distance) / 6371000 #Earth Radius

        # Calculate the latitude and longitude offsets
        lat_offset = random_distance * math.cos(random_direction)
        lon_offset = random_distance * math.sin(random_direction)

        # Calculate the new latitude and longitude
        new_lat = lat + lat_offset
        new_lon = lon + lon_offset

        return new_lat, new_lon

    ########################################################################################
    # Execute the main engine loop (which waits for messages to arrive from MQTT)
    ########################################################################################
    def execute(self):
        print("Engine started.")
        client = paho.Client()
        client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message
        
        # retry connection until this succeeds
        connected = False
        while not connected:
            try:
                client.connect(self.config['MQTT_CLIENT_URL'], self.config['MQTT_CLIENT_PORT'])
                connected=True
            except:
                time.sleep(1)    
        
        print(f'Subscribing to MQTT: {self.config["MQTT_CLIENT_URL"]} {self.config["MQTT_PUBLISH_URL"]}')
        client.subscribe(self.config['MQTT_PUBLISH_URL'])
        
        print("Retrieving species names from GCP")
        self.class_names = self.gcp_load_species_list()
        
        for cs in self.class_names:
            print(f" class name {cs}")

        print("Engine waiting for audio to arrive...")
        client.loop_forever()

    


if __name__ == "__main__":
    
    engine = EchoEngine()
    engine.execute()
