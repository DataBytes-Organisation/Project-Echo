from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime
import os

connection_string = os.getenv("MONGODB_URI")

if not connection_string:
    raise RuntimeError("MONGODB_URI environment variable is required for the connection test.")

client = MongoClient(connection_string)
db = client["EchoNet"]

# Insert a new event
event = {
    "timestamp": datetime.datetime.utcnow(),
    "sensorID": 123,
    "microphoneLLA": [0.0, 0.0, 0.0],
    "animalEstLLA": [0.0, 0.0, 0.0],
    "animalTrueLLA": [0.0, 0.0, 0.0],
    "animalLLAUncertainty": 0.0,
    "audioClip": "base64_audio_data",
    "confidence": 0.95,
}

db.events.insert_one(event)

# Insert a new species
species = {
    "scientificName": "Canis lupus",
    "commonName": "gray wolf",
    "type": "mammal",
    "status": "least concern",
    "diet": "carnivore",
}

db.species.insert_one(species)

# Insert a new movement
movement = {
    "timestamp": datetime.datetime.utcnow(),
    "animalId": 1,
    "speciesId": 1,
    "animalTrueLLA": [0.0, 0.0, 0.0],
}

db.movements.insert_one(movement)
