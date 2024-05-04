from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator, constr
from bson.objectid import ObjectId
from app.database import GENDER, STATES_CODE, AUS_STATES

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


class AddressSchema(BaseModel):
    country: str
    state: Optional[str]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 
class RequestSchema(BaseModel):
    
    #requestId: str
    username: str
    animal: str
    # requestingToChange: str
    initial: str
    modified: str
    source: str
    summary: str
    desc: str
    date: datetime
    status: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 

        
class UserSignupSchema(BaseModel):
    username: str
    password: constr(min_length = 8)
    email: str
    roles: List[str]
    gender: str
    DoB: datetime
    address: AddressSchema
    organization: str
    phonenumber: Optional[str]


    @validator('DoB', pre=True)
    def validate_dob(cls, value: object) -> object:
        try:
            if isinstance(value, str):
                return datetime.combine(datetime.strptime(value, '%d/%m/%Y'), datetime.min.time())
        except ValueError:
            raise ValueError('Invalid date format. Please use "dd/mm/yyyy".')
    
    @validator('address')
    def validate_address(cls, value):
        try:
            country = value.country
        
            if country.lower() == "australia" or country.lower() == "aus":
                state = value.state
                if state is None:    
                    raise ValueError('State is required')
            
                if state.lower() not in STATES_CODE + AUS_STATES:
                    raise ValueError('State does not exist')
                value.state = state.upper()
            return value
                
        except ValueError as e:
            raise ValueError('Error in Address Field:' + str(e))

    @validator('gender')
    def validate_gender(cls, value):
        if value.lower() not in GENDER:
            raise ValueError('Please select the available gender.')
        return value.upper()


    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 

   
class UserLoginSchema(BaseModel):
    username: str
    email: str
    password: str
    

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 
class GuestSchema(BaseModel):
    username: str
    email: str
    password: str
    userId: str
    roles: List[dict]
    expiresAt: datetime

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 

class GuestSignupSchema(BaseModel):
    username: str
    email: str
    password: str
    timestamp: datetime

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 

class ForgotPasswordSchema(BaseModel):
    user: str
    

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 

class RecordingData(BaseModel):
    timestamp: datetime
    sensorId: str
    microphoneLLA: List[float] 
    animalEstLLA: List[float]
    animalTrueLLA: List[float]
    animalLLAUncertainty: float
    audioClip: str
    mode: str 
    audioFile: str
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 