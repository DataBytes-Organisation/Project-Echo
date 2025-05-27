from datetime import datetime
from typing import Optional, List, Dict, Annotated
from pydantic import BaseModel, Field, validator, constr, condecimal
from bson.objectid import ObjectId
from app.database import GENDER, STATES_CODE, AUS_STATES

# Custom class to validate MongoDB ObjectId
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


# Event schema
class EventSchema(BaseModel):
    timestamp: datetime
    sensorId: constr(min_length=1)
    species: constr(min_length=1)
    microphoneLLA: Annotated[List[float], Field(min_items=3, max_items=3)]
    animalEstLLA: Annotated[List[float], Field(min_items=3, max_items=3)]
    animalTrueLLA: Annotated[List[float], Field(min_items=3, max_items=3)]
    animalLLAUncertainty: int
    audioClip: str
    confidence: condecimal(gt=0, lt=100)
    sampleRate: int

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "timestamp": "2023-03-22T13:45:12.000Z",
                "sensorId": "2",
                "species": "Sus Scrofa",
                "microphoneLLA": [-33.1101, 150.0567, 23],
                "animalEstLLA": [-33.1105, 150.0569, 23],
                "animalTrueLLA": [-33.1106, 150.0570, 23],
                "animalLLAUncertainty": 10,
                "audioClip": "some audio_base64 data",
                "confidence": 99.4,
                "sampleRate": 48000
            }
        }


# Movement schema
class MovementSchema(BaseModel):
    timestamp: datetime
    species: constr(min_length=1)
    animalId: constr(min_length=1)
    animalTrueLLA: Annotated[List[float], Field(min_items=3, max_items=3)]

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "timestamp": "2023-03-22T13:45:12.000Z",
                "species": "Sus Scrofa",
                "animalId": "1",
                "animalTrueLLA": [-33.1106, 150.0570, 23]
            }
        }


# Microphone schema
class MicrophoneSchema(BaseModel):
    sensorId: constr(min_length=1)
    microphoneLLA: Annotated[List[float], Field(min_items=3, max_items=3)]

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "sensorId": "2",
                "microphoneLLA": [-33.1106, 150.0570, 23]
            }
        }


# Address schema
class AddressSchema(BaseModel):
    country: str
    state: Optional[str]

    @validator('state', pre=True, always=True)
    def validate_state(cls, value, values):
        if 'country' in values and values['country'].lower() in ["australia", "aus"]:
            if not value:
                raise ValueError("State is required for Australia")
            if value.lower() not in STATES_CODE + AUS_STATES:
                raise ValueError(f"{value} is not a valid state for Australia")
        return value

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request schema
class RequestSchema(BaseModel):
    username: str
    animal: str
    requestingToChange: str
    initial: str
    modified: str
    source: str
    date: datetime
    status: str

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Components
class ComponentSchema(BaseModel):
    type: str
    model: str
    customProperties: Dict[str, str]


class LocationSchema(BaseModel):
    latitude: float
    longitude: float


class NodeSchema(BaseModel):
    id: str  # changed from "_id" to "id"
    name: str
    type: str
    model: str
    location: LocationSchema
    customProperties: Dict[str, str]
    components: List[ComponentSchema] = []
    connectedNodes: List[str] = []

    class Config:
        schema_extra = {
            "example": {
                "id": "node001",
                "name": "Alpha Node",
                "type": "Sensor",
                "model": "Model-X1",
                "location": {
                    "latitude": -33.1234,
                    "longitude": 150.5678
                },
                "customProperties": {
                    "region": "North",
                    "active": "true"
                },
                "components": [
                    {
                        "type": "Microphone",
                        "model": "MicPro",
                        "customProperties": {
                            "sensitivity": "high"
                        }
                    }
                ],
                "connectedNodes": []
            }
        }

# Animal notification
class AnimalNotificationSchema(BaseModel):
    species: str
    confidence: float
    common: str

    class Config:
        json_schema_extra = {
            "example": {
                "species": "Sus scrofa",
                "common": "Wild Boar"
            }
        }


# Email notification
class EmailNotifications(BaseModel):
    species: str
    last_sent: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "species": "Sus scrofa",
                "last_sent": "2024-09-15T10:30:00Z"
            }
        }


# User signup
class UserSignupSchema(BaseModel):
    username: str
    password: constr(min_length=8)
    email: str
    roles: List[str]
    gender: str
    DoB: datetime
    address: AddressSchema
    organization: str
    phonenumber: Optional[str]
    notificationAnimals: Optional[List[AnimalNotificationSchema]] = []
    emailNotifications: Optional[List[EmailNotifications]] = []

    @validator('DoB', pre=True)
    def validate_dob(cls, value):
        if isinstance(value, str):
            return datetime.combine(datetime.strptime(value, '%d/%m/%Y'), datetime.min.time())
        return value

    @validator('address')
    def validate_address(cls, value):
        country = value.country
        if country.lower() in ["australia", "aus"]:
            state = value.state
            if not state:
                raise ValueError('State is required for Australia')
            if state.lower() not in STATES_CODE + AUS_STATES:
                raise ValueError('State does not exist')
            value.state = state.upper()
        return value

    @validator('gender')
    def validate_gender(cls, value):
        if value.lower() not in GENDER:
            raise ValueError('Please select the available gender.')
        return value.upper()

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Login
class UserLoginSchema(BaseModel):
    username: str
    email: str
    password: str

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Guest schema
class GuestSchema(BaseModel):
    username: str
    email: str
    password: str
    userId: str
    roles: List[dict]
    expiresAt: datetime

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class GuestSignupSchema(BaseModel):
    username: str
    email: str
    password: str
    timestamp: datetime

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Forgot/reset password
class ForgotPasswordSchema(BaseModel):
    user: str

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ResetPasswordSchema(BaseModel):
    user: str
    password: str
    otp: int

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Recording schema
class RecordingData(BaseModel):
    timestamp: datetime
    sensorId: str
    microphoneLLA: Annotated[List[float], Field(min_items=3, max_items=3)]
    animalEstLLA: Annotated[List[float], Field(min_items=3, max_items=3)]
    animalTrueLLA: Annotated[List[float], Field(min_items=3, max_items=3)]
    animalLLAUncertainty: condecimal(gt=0)
    audioClip: str
    mode: str
    audioFile: str

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
