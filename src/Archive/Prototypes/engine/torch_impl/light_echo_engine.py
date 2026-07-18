
##################################################################################################
# This is the Light Echo Engine application
#
# This program assumes you have already trained a model TfLite or ONNX format
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
        # STEP 1: Convert to tensor
        pred_tensor = tf.convert_to_tensor(predictions)
        
        # STEP 2: SCALE THE LOGITS
        # Check your training config for the exact value of 's'.
        # CircleLoss parameters
        s = 80.0
        m = 0.4
        delta_n = m 
        
        # Apply CircleLoss transformation
        scaled_predictions = s * (pred_tensor - delta_n)
        
        # Get prediction
        predicted_index = int(tf.argmax(tf.squeeze(scaled_predictions)).numpy())
        
        # STEP 3: Safe Lookup
        if hasattr(self, 'class_names') and self.class_names and predicted_index < len(self.class_names):
            predicted_class = self.class_names[predicted_index]
        else:
            predicted_class = f"Class_Index_{predicted_index}"
            print(f"WARNING: Index {predicted_index} not found in class list.")

        # STEP 4: Calculate Probability
        predicted_probability = 100.0 * tf.nn.softmax(scaled_predictions)[predicted_index].numpy()
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


    def combined_pipeline(self, audio_clip, mode):
        """
            Complete audio processing pipeline to convert raw audio bytes into a model-ready spectrogram image.

            Args:
                audio_clip (bytes): The raw audio clip in bytes
        """
        
        sample_rate = 0
        file = io.BytesIO(audio_clip)
        
        # Load audio
        if str(mode) in ["Recording_Mode", "Recording_Mode_V2"]:
            audio_clip, sample_rate = librosa.load(file, sr=self.config['AUDIO_SAMPLE_RATE'])
        else:
            audio_clip, sample_rate = librosa.load(file, sr=self.config['AUDIO_SAMPLE_RATE'])

        # Keep right channel only
        if audio_clip.ndim == 2 and audio_clip.shape[0] == 2:
            audio_clip = audio_clip[1, :]
        
        # Cast to float32
        audio_clip = audio_clip.astype(np.float32)
        
        # Extract random subsection
        audio_clip = self.load_random_subsection(audio_clip, duration_secs=self.config['AUDIO_CLIP_DURATION'])
        
        # STEP 1: Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_clip, 
            sr=self.config['AUDIO_SAMPLE_RATE'], 
            n_fft=self.config['AUDIO_NFFT'], 
            hop_length=self.config['AUDIO_STRIDE'], 
            n_mels=self.config['AUDIO_MELS'],
            fmin=self.config['AUDIO_FMIN'],
            fmax=self.config['AUDIO_FMAX'],
            win_length=self.config['AUDIO_WINDOW']
        )

        # STEP 2: Convert to dB (CRITICAL FIX)
        image = librosa.power_to_db(mel_spec, top_db=self.config['AUDIO_TOP_DB'])
        image = np.clip(image, -80.0, 0.0)

        # STEP 3: Reshape and Resize
        expected_clip_samples = int(self.config['AUDIO_CLIP_DURATION'] * self.config['AUDIO_SAMPLE_RATE'] / self.config['AUDIO_STRIDE'])
        image = np.moveaxis(image, 1, 0)
        image = image[0:expected_clip_samples, :]

        # Add channel dimension
        image = tf.expand_dims(image, -1)

        # For 3-channel models, repeat
        if self.config['MODEL_INPUT_IMAGE_CHANNELS'] == 3:
            image = tf.repeat(image, 3, axis=2)

        # Ensure shape
        image = tf.ensure_shape(image, [expected_clip_samples, self.config['AUDIO_MELS'], self.config['MODEL_INPUT_IMAGE_CHANNELS']])
        
        # Resize to 224x224
        image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.LANCZOS5)
        
        # STEP 4: Normalize to [0, 1] with FIXED scale
        image = (image + 80.0) / 80.0
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        # STEP 5: Apply ImageNet Normalization
        if self.config['MODEL_INPUT_IMAGE_CHANNELS'] == 1:
            mean_val = 0.449
            std_val = 0.226
            image = (image - mean_val) / std_val
            
        elif self.config['MODEL_INPUT_IMAGE_CHANNELS'] == 3:
            mean_rgb = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            std_rgb = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
            image = (image - mean_rgb) / std_rgb
        

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

                image = tf.transpose(image, perm=[0, 3, 1, 2])  # [1, 1, 224, 224]

                image_list = image.numpy().tolist()

                # Run the model via tensorflow serve
                data = json.dumps({"signature_name": "serving_default", "inputs": image_list})
                url = self.config['MODEL_SERVER']
                headers = {"content-type": "application/json"}
                json_response = requests.post(url, data=data, headers=headers)
                
                model_result = json.loads(json_response.text)

                # Then handle different response formats:
                if 'outputs' in model_result:
                    predictions = model_result['outputs'][0]
                elif 'error' in model_result:
                    print(f"[ERROR] Server error: {model_result['error']}")
                    raise Exception(model_result['error'])
                else:
                    print(f"[ERROR] Unexpected format: {model_result}")
                    raise KeyError(f"Available keys: {list(model_result.keys())}")
                    
                # Predict class and probability using the prediction function
                predicted_class, predicted_probability = self.predict_class(predictions)

                print(f'Predicted class : {predicted_class}')
                print(f'Predicted probability : {predicted_probability}')
            
                # populate the database with the result
                self.echo_api_send_detection_event(
                    audio_event,
                    sample_rate,
                    predicted_class,
                    predicted_probability)
                    
            
                image = tf.expand_dims(image, 0) 
            
                image_list = image.numpy().tolist()

            else: # simulate animals mode
                # convert to string representation of audio to binary for processing
                audio_clip = self.string_to_audio(audio_event['audioClip'])
            
                image, audio_clip, sample_rate = self.combined_pipeline(audio_clip, "Animal_Mode")
            
                #returned is melspectrogram with cam overlay,
                #TODO: Can add this image to database
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
                    predicted_probability)
            
                image = tf.expand_dims(image, 0) 
            
                image_list = image.numpy().tolist()
            
        except Exception as e:
            # Catch the exception and print it to the console
            print(f"An error occurred: {e}", flush=True)


    ########################################################################################
    # this function populates the database with the prediction results
    ########################################################################################
    def echo_api_send_detection_event(self, audio_event, sample_rate, predicted_class, predicted_probability):
        
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
            "sampleRate": sample_rate     
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

    def predict_weather_audio(self, audio_clip):
        """
        Call with audio clip, it will call weather detection model running on model container
        """
        image, audio, sample_rate = self.weather_pipeline(audio_clip)

        image = tf.expand_dims(image, 0) 
            
        image_list = image.numpy().tolist()

        data = json.dumps({"signature_name": "serving_default", "inputs": image_list})
        url = self.config['WEATHER_SERVER']
        headers = {"content-type": "application/json"}
        json_response = requests.post(url, data=data, headers=headers)
        model_result   = json.loads(json_response.text)
        predictions = model_result['outputs'][0]
        print("Weather Prediciton", predictions)
        #TODO: Map prediciton of appropriate class label
        return predictions


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
        print("Engine started.", flush=True)
        # Use simple client setup for compatibility
        client = paho.Client()
        client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message
        
        print(f"DEBUG: Attempting to connect to Broker at: {self.config['MQTT_CLIENT_URL']}:{self.config['MQTT_CLIENT_PORT']}", flush=True)

        connected = False
        while not connected:
            try:
                client.connect(self.config['MQTT_CLIENT_URL'], self.config['MQTT_CLIENT_PORT'])
                connected=True
                print("DEBUG: Connection Successful!", flush=True)
            except Exception as e:
                # PRINT THE ERROR SO WE CAN SEE IT
                print(f"CONNECTION ERROR: {e}", flush=True)
                time.sleep(2)
        
        print(f'Subscribing to MQTT: {self.config["MQTT_CLIENT_URL"]} {self.config["MQTT_PUBLISH_URL"]}')
        client.subscribe(self.config['MQTT_PUBLISH_URL'])
        
        print("Retrieving species names from GCP")
        self.class_names = self.gcp_load_species_list()
        
        # ------------------------------------------
        for cs in self.class_names:
            print(f" class name {cs}")

        print("Engine waiting for audio to arrive...")
        client.loop_forever()

    


if __name__ == "__main__":
    
    engine = EchoEngine()
    engine.execute()
