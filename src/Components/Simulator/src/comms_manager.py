

#############################################################################
# This class provides message handling for read and writing JSON messages
# This class interfaces with google cloud for audio data
# This class interfaces with mqtt for message passing
#############################################################################

import paho.mqtt.client as paho
import base64
import json
import pymongo
from google.cloud import storage
import os
from entities.species import Species
import random
from clock import Clock
import logging
import datetime
import requests

logger1 = logging.getLogger('_sys_logger')
        
class CommsManager():
    
    def __init__(self) -> None:
        self.audio_blobs = {}
        self.clock = Clock()
       
    # Initialise communication with MQTT endpoints
    def initialise_communications(self):
        
        logger1.info(f'Initialising Communications')
        
        self.mqtt_client = paho.Client()
        self.mqtt_client.connect(os.environ['MQTT_CLIENT_URL'], int(os.environ['MQTT_CLIENT_PORT']))
       
        # Load the project echo credentials into a dictionary
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_credentials.json')
            with open(file_path, 'r') as f:
                self.credentials = json.load(f)
            print(f"Echo Simulator credentials successfully loaded", flush=True)
        except:
            print(f"Could not load Echo Simulator credentials : {file_path}") 
                   
        # Setup database client and connect
        try:
            # database connection string
            self.connection_string=f"{os.getenv('ATLAS_CONNECTION_STRING')}"

            myclient = pymongo.MongoClient(self.connection_string)
            self.echo_store = myclient["EchoNet"]
            print(f"Found echo store database names: {myclient.list_database_names()}", flush=True)
        except:
            print(f"Failed to establish database connection", flush=True)


    # This function uses the google bucket with audio files and
    # leverages the folder names as the official species names
    # Note: to run this you will need to first authenticate
    # See https://github.com/DataBytes-Organisation/Project-Echo/tree/main/src/Prototypes/data#readme
    def gcp_load_species_list(self):
 
        species_names = set()

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(os.environ['BUCKET_NAME'])
        blobs = bucket.list_blobs()  # Get list of files
        for blob in blobs:
            folder_name = blob.name.split('/')[0]
       
            species_names.add(folder_name)
            
            if folder_name in self.audio_blobs:
                self.audio_blobs[folder_name].append(blob)
            else:
                self.audio_blobs[folder_name] = []
                self.audio_blobs[folder_name].append(blob)
        
        # using the names loaded from GCP, construct the Species objects
        species_list = []
        for name in species_names:
            species = Species(name)
            species_list.append(species)
            
        return species_list
 
    # send a random audio message for the given animal at the predicted lla
    def mqtt_send_random_audio_msg(self, animal, predicted_lla, closest_mic, min_error) -> None:
        
        # get the timestamp for this event
        timestamp = self.clock.get_time()
        
        # get the species name
        species_name    = animal.getSpecies().getName()
        animal_true_lla = animal.getLLA()
        
        # microphone LLA TODO
        microphone_lla  = closest_mic.getLLA()
        
        # randomly sample from available audio blobs from this species
        sample_blob = random.sample(self.audio_blobs[species_name], k=1)[0]
        
        # Read the blob's content as a byte array
        audio = sample_blob.download_as_bytes()
 
        # Encode the audio data as a string
        audio_str = self.audio_to_string(audio)
        
        # For now, send filename across for format information
        audio_file = sample_blob.name.split('/')[1]
         
        # Create the vocalisation event
        vocalisation_event = {
            "timestamp": timestamp.isoformat(),
            "sensorId": closest_mic.getID(),
            "microphoneLLA": list(microphone_lla),
            "animalEstLLA": list(predicted_lla),
            "animalTrueLLA": list(animal_true_lla),
            "animalLLAUncertainty": min_error,
            "audioClip" : audio_str,
            "mode" : "Animal_Mode", 
            "audioFile" : audio_file      
        }    
        
        MQTT_MSG = json.dumps(vocalisation_event)
     
        # publish the audio message on the queue
        (rc, mid) = self.mqtt_client.publish(os.environ['MQTT_PUBLISH_URL'], MQTT_MSG)
        
        logger1.info(f'Vocal message sent {animal.getUUID()} time: {timestamp} species: {species_name}')
        print(f'Vocal message sent {animal.getUUID()} time: {timestamp} species: {species_name}', flush=True)

        # send a random audio message for the given animal at the predicted lla
    def mqtt_send_recording_msg(self, msg, mode) -> None:
        
        #print(msg.payload)

        # publish the audio message on the queue
        (rc, mid) = self.mqtt_client.publish(os.environ['MQTT_PUBLISH_URL'], msg.payload)
        
        logger1.info(f'Recording message sent')
        print(f'Recording message sent', flush=True)


    ########################################################################################
    # this function populates the database with animal movement events
    ########################################################################################
    def echo_api_send_animal_movement(self, animal):

        movement_event = {
            "timestamp": self.clock.get_time().timestamp(),
            "species": animal.getSpecies().getName(),
            "animalId": animal.getUUID(),
            "animalTrueLLA": list(animal.getLLA())    
        }

        url = 'http://ts-api-cont:9000/sim/movement'

        x = requests.post(url, json = movement_event)

        print(x.text)
        

    ########################################################################################
    # this function populates the database with all the microphones
    ########################################################################################        
    def echo_api_set_microphones(self, microphones):
        
        microphone_list = []
        
        for mic in microphones:
            lla = mic.getLLA()
            microphone = {
                "sensorId": mic.getID(),
                "microphoneLLA": [
                    lla[0],
                    lla[1],
                    lla[2]
                ]
            }
            print(f'Setting Mic {microphone}')
            microphone_list.append(microphone)
 
        url = 'http://ts-api-cont:9000/sim/microphones'
        x = requests.post(url, json = microphone_list)
        print(x.text)

    # this method takes in binary audio data and encodes to string
    def audio_to_string(self, audio_binary) -> str:
        base64_encoded_data = base64.b64encode(audio_binary)
        base64_message = base64_encoded_data.decode('utf-8')
        return base64_message
    
    def string_to_audio(self, audio_string) -> bytes:
        base64_img_bytes = audio_string.encode('utf-8')
        decoded_data = base64.decodebytes(base64_img_bytes)
        return decoded_data
    
    def test(self):
        logger1.info(f'testing MessageManager')
        
        logger1.info(f'Testing GCP endpoint')
        
        species_list = self.gcp_load_species_list()
        for species in species_list:
            logger1.info(f'Found species : {species.getName()}')
        
        # load a test json file containing audio data
        with open('src\Prototypes\data\database\sample_data\events.json', 'r') as file:
            test_json = json.load(file)
            msg = test_json[0]
            logger1.info(f' Loaded message timestamp: {msg["timestamp"]}')
            
            audio_b1 = self.string_to_audio(msg['audioClip'])
            audio_s1 = self.audio_to_string(audio_b1)
            
            audio_b2 = self.string_to_audio(audio_s1)
            audio_s2 = self.audio_to_string(audio_b2)
            
            audio_b3 = self.string_to_audio(audio_s2)
            audio_s3 = self.audio_to_string(audio_b3)
            
            assert audio_s3 == audio_s1, "Strings are not matching!"
            
            logger1.info(f'test completed successfully')
    
        