def eventEntity(event) -> dict:
    return {
        "id": str(event["_id"]),
        "timestamp": event["timestamp"],
        "sensorId": event["sensorId"],
        "species": event["species"],
        "microphoneLLA": event["microphoneLLA"],
        "animalEstLLA": event["animalEstLLA"],
        "animalTrueLLA": event["animalTrueLLA"],
        "animalLLAUncertainty": event["animalLLAUncertainty"],
        "audioClip": event["audioClip"],
        "confidence": event["confidence"]  
    }

def eventListEntity(events) -> list:
    return [eventEntity(event) for event in events]
