def eventEntity(event) -> dict:
    return {
        "_id": str(event["_id"]),
        "timestamp": event["timestamp"],
        "sensorId": event["sensorId"],
        "species": event["species"],
        "microphoneLLA": event["microphoneLLA"],
        "animalEstLLA": event["animalEstLLA"],
        "animalTrueLLA": event["animalTrueLLA"],
        "animalLLAUncertainty": event["animalLLAUncertainty"],
        "confidence": event["confidence"],
    }

def eventSpeciesEntity(event) -> dict:
    return {
        "_id": str(event["_id"]),
        "commonName": event["commonName"],
        "type": event["type"],
        "status": event["status"],
        "diet": event["diet"],
        "timestamp": event["timestamp"],
        "sensorId": event["sensorId"],
        "species": event["species"],
        "microphoneLLA": event["microphoneLLA"],
        "animalEstLLA": event["animalEstLLA"],
        "animalTrueLLA": event["animalTrueLLA"],
        "animalLLAUncertainty": event["animalLLAUncertainty"],
        "confidence": event["confidence"],
    }

def movementEntity(event) -> dict:
    return {
        "_id": str(event["_id"]),
        "timestamp": event["timestamp"],
        "species": event["species"],
        "animalId": event["animalId"],
        "animalTrueLLA": event["animalTrueLLA"]
    }

def movementSpeciesEntity(event) -> dict:
    return {
        "_id": str(event["_id"]),
        "commonName": event["commonName"],
        "type": event["type"],
        "status": event["status"],
        "diet": event["diet"],
        "timestamp": event["timestamp"],
        "species": event["species"],
        "animalId": event["animalId"],
        "animalTrueLLA": event["animalTrueLLA"]
    }

def speciesEntity(species) -> dict:
    return {
        "_id": str(species["_id"]),
        "commonName": species["commonName"],
        "type": species["type"],
        "status": species["status"],
        "diet": species["diet"]
    }

def audioEntity(audio) -> dict:
    return {
        "_id": str(audio["_id"]),
        "audioClip": audio["audioClip"],
        "sampleRate": audio["sampleRate"]
    }
    
def microphoneEntity(microphone) -> dict:
    return{
        "_id": str(microphone["_id"]),
        "microphoneLLA": microphone["microphoneLLA"]
    }
def timestampEntity(timestamp) -> dict:
    return{
        "timestamp": timestamp["timestamp"]
    }
    
def eventListEntity(events) -> list:
    return [eventEntity(event) for event in events]
    
def eventSpeciesListEntity(events) -> list:
    return [eventSpeciesEntity(event) for event in events]

def audioListEntity(audios) -> list:
    return [audioEntity(audio) for audio in audios]

def movementListEntity(events) -> list:
    return [movementEntity(event) for event in events]

def movementSpeciesListEntity(events) -> list:
    return [movementSpeciesEntity(event) for event in events]

def microphoneListEntity(microphones) -> list:
    return [microphoneEntity(microphone) for microphone in microphones]

def timestampListEntity(timestamps) -> list:
    return [timestampEntity(timestamp) for timestamp in timestamps]
