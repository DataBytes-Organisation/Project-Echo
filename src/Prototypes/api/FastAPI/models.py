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

class EventModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    timestamp: datetime
    sensorId: int
    species: str 
    microphoneLLA: List[float] 
    animalEstLLA: List[float]
    animalTrueLLA: List[float]
    animalLLAUncertainty: int 
    audioClip: str 
    confidence: float

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {                          
                "timestamp": "2023-03-22T13:45:12.000Z",
                "sensorId": 2,
                "species": "Sus Scrofa",
                "microphoneLLA":[-33.1101,150.0567, 23],
                "animalEstLLA": [-33.1105,150.0569, 23],
                "animalTrueLLA": [-33.1106,150.0570, 23],
                "animalLLAUncertainty": 10,
                "audio clip": "some audio_base64 data",
                "confidence": 99.4
            }
        }
'''
class speciesModel(BaseModel):
    return {
        "_id": str(species["_id"]),
        "commonName": species["commonName"],
        "type": species["type"],
        "status": species["status"],
        "diet": species["diet"]
    }
'''