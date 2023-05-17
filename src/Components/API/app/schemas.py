from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from bson.objectid import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class EventSchema(BaseModel):
    timestamp: datetime
    sensorId: str
    species: str 
    microphoneLLA: List[float] 
    animalEstLLA: List[float]
    animalTrueLLA: List[float]
    animalLLAUncertainty: int 
    audioClip: str 
    confidence: float
    sampleRate: int

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {                          
                "timestamp": "2023-03-22T13:45:12.000Z",
                "sensorId": "2",
                "species": "Sus Scrofa",
                "microphoneLLA":[-33.1101,150.0567, 23],
                "animalEstLLA": [-33.1105,150.0569, 23],
                "animalTrueLLA": [-33.1106,150.0570, 23],
                "animalLLAUncertainty": 10,
                "audioClip": "some audio_base64 data",
                "confidence": 99.4,
                "sampleRate": 48000
            }
        }
class MovementSchema(BaseModel):
    timestamp: datetime
    species: str
    animalId: str
    animalTrueLLA: List[float]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {                          
                "timestamp": "2023-03-22T13:45:12.000Z",
                "species": "Sus Scrofa",
                "animalId": "1",
                "animalTrueLLA": [-33.1106,150.0570, 23]
            }
        }

class MicrophoneSchema(BaseModel):
    sensorId: str
    microphoneLLA: List[float] 

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                    "sensorId": "2",
                    "microphoneLLA": [-33.1106,150.0570, 23],           
            }
        }