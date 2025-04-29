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
        "commonName": event.get("commonName", "Unknown"),
        "type": event.get("type", "Unknown"),
        "status": event.get("status", "Unknown"),
        "diet": event.get("diet", "Unknown"),
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
        "commonName": event.get("commonName", "Unknown"),
        "type": event.get("type", "Unknown"),
        "status": event.get("status", "Unknown"),
        "diet": event.get("diet", "Unknown"),
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

def requestEntity(request) -> dict:
    return {
        "_id": str(request["_id"]),
        #"requestId": str(request["requestId"]),
        "username": str(request["username"]),
        "animal": str(request["animal"]),
        "requestingToChange": str(request["requestingToChange"]),
        "initial": str(request["initial"]),
        "modified": str(request["modified"]),
        "source": str(request["source"]),
        "date": str(request["date"]),
        "status": str(request["status"])
    }

# Updated userEntity to include visits and totalTime
def userEntity(user) -> dict:
    return{
        "username": user["username"],
        "visits": user.get("visits", 0),  # Include visits, default to 0 if not present
        "totalTime": user.get("totalTime", 0)  # Include totalTime, default to 0 if not present
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

# Fixed typo in userListEntity (username -> users, userName -> user)
def userListEntity(users) -> list:
    return [userEntity(user) for user in users]

def requestListEntity(requests) -> list:
    return [requestEntity(request) for request in requests]

# converting cursor to dict for get request of animals data
def animalEntity(animal) -> dict:
    if len(animal.keys())>5:
        return {
            "species": animal["_id"],
            "commonName": (animal["commonName"]),
            "type" : (animal["type"]),
            "status" : (animal["status"]),
            "diet" : (animal["diet"]),
            "timestamp": animal["timestamp"],
            "sensorId": animal["sensorId"],
            "microphoneLLA": str(animal["microphoneLLA"]),
            "animalEstLLA": str(animal["animalEstLLA"]),
            "animalTrueLLA": str(animal["animalTrueLLA"]),
            "animalLLAUncertainty": animal["animalLLAUncertainty"],
            "audioClip" : str(animal["audioClip"]),
            "confidence": animal["confidence"],
            "sampleRate" : animal["sampleRate"]
        }
    else:
        return {
            "species": animal["_id"],
            "commonName": (animal["commonName"]),
            "type" : (animal["type"]),
            "status" : (animal["status"]),
            "diet" : (animal["diet"]),
            "timestamp": "",
            "sensorId": "",
            "microphoneLLA": "",
            "animalEstLLA": "",
            "animalTrueLLA": "",
            "animalLLAUncertainty": "",
            "audioClip" : "",
            "confidence": "",
            "sampleRate" : ""
        }

def animalListEntity(animals) -> list:
    return [animalEntity(animal) for animal in animals]