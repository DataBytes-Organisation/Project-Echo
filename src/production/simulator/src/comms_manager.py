

#############################################################################
# This class provides message handling for read and writing JSON messages
# This class interfaces with Cloudflare R2 for audio data
# This class interfaces with mqtt for message passing
#############################################################################

import paho.mqtt.client as paho
import base64
import json
import pymongo
import os
from pathlib import PurePosixPath
from r2_storage import R2Storage
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
        self.r2_storage = None
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
            self.connection_string=f"mongodb://{self.credentials['DB_USERNAME']}:{self.credentials['DB_PASSWORD']}@{os.environ['MONGODB_HOSTNAME']}/EchoNet"

            myclient = pymongo.MongoClient(self.connection_string)
            self.echo_store = myclient["EchoNet"]
            print(f"Found echo store database names: {myclient.list_database_names()}", flush=True)
        except:
            print(f"Failed to establish database connection", flush=True)


    # Load species and audio object keys from Cloudflare R2.
    def r2_load_species_list(self):
        self.r2_storage = R2Storage()
        self.audio_blobs = self.r2_storage.list_audio_by_species()

        species_list = []
        for name in sorted(self.audio_blobs):
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
        sample_key = random.sample(self.audio_blobs[species_name], k=1)[0]
        
        # Read the blob's content as a byte array
        audio = self.r2_storage.download_bytes(sample_key)
 
        # Encode the audio data as a string
        audio_str = self.audio_to_string(audio)
        
        # For now, send filename across for format information
        audio_file = PurePosixPath(sample_key).name

        print("DEBUG audio_file:", audio_file, flush=True)
        print("DEBUG audio bytes length:", len(audio), flush=True)
        print("DEBUG first 20 bytes:", audio[:20], flush=True)
        print("DEBUG audioClip base64 length:", len(audio_str), flush=True)
        print("DEBUG audio_file extension:", audio_file.split(".")[-1], flush=True)
                
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
        
        logger1.info(f'Testing Cloudflare R2 endpoint')
        
        species_list = self.r2_load_species_list()
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
    
