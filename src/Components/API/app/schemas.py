from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator, constr, conlist, condecimal
from bson.objectid import ObjectId
from app.database import GENDER, STATES_CODE, AUS_STATES

# Custom Pydantic Validator for ObjectId (MongoDB)
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

class EventTimeSchema(BaseModel):
    start: datetime  # Automatically parses ISO 8601 date strings into datetime objects
    end: datetime    # Automatically parses ISO 8601 date strings into datetime objects

    class Config:
        schema_extra = {
            "example": {
                "start": "2023-01-01T00:00:00",
                "end": "2023-01-02T00:00:00"
            }
        }

# Event Schema for input validation (with comments)
class EventSchema(BaseModel):
    timestamp: datetime  # Timestamp for the event
    sensorId: constr(min_length=1)  # Sensor ID must be a non-empty string
    species: constr(min_length=1)  # Species name must be a non-empty string
    microphoneLLA: conlist(float, min_items=3, max_items=3)  # List of exactly 3 floats for microphone location
    animalEstLLA: conlist(float, min_items=3, max_items=3)  # Estimated location of the animal
    animalTrueLLA: conlist(float, min_items=3, max_items=3)  # True location of the animal
    animalLLAUncertainty: int  # Integer to represent uncertainty level
    audioClip: str  # Audio clip in base64 format
    confidence: condecimal(gt=0, lt=100)  # Confidence level must be between 0 and 100
    sampleRate: int  # Sample rate for audio

    class Config:
        populate_by_name = True
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

# Movement Schema for validation of movement data
class MovementSchema(BaseModel):
    timestamp: datetime  # Timestamp for the movement
    species: str  # Species involved in the movement
    animalId: str  # ID of the animal
    animalTrueLLA: conlist(float, min_items=3, max_items=3)  # True location of the animal

    class Config:
        populate_by_name = True
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

# Microphone Schema to handle input for microphone data
class MicrophoneSchema(BaseModel):
    sensorId: str  # Sensor ID for the microphone
    microphoneLLA: conlist(float, min_items=3, max_items=3)  # Microphone location (latitude, longitude, altitude)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "sensorId": "2",
                "microphoneLLA": [-33.1106, 150.0570, 23],           
            }
        }

# Address Schema used in UserSignupSchema for validating address
class AddressSchema(BaseModel):
    country: str  # Country field
    state: Optional[str]  # State field is optional, but required for specific countries

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @validator('state', pre=True, always=True)
    def validate_state(cls, value, values):
        if 'country' in values and values['country'].lower() in ["australia", "aus"]:
            if not value:
                raise ValueError("State is required for Australia")
            if value.lower() not in STATES_CODE + AUS_STATES:
                raise ValueError(f"{value} is not a valid state for Australia")
        return value

# Request Schema for handling requests made to change animal information
class RequestSchema(BaseModel):
    username: str  # Requestor's username
    animal: str  # Animal identifier
    requestingToChange: str  # Field being requested to change
    initial: str  # Initial value before change
    modified: str  # Modified value after request
    source: str  # Source of the request
    date: datetime  # Date of the request
    status: str  # Request status (pending, completed, etc.)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# AnimalNotificationSchema for user notifications about animals
class AnimalNotificationSchema(BaseModel):
    species: str
    common: str

    class Config:
        schema_extra = {
            "example": {
                "species": "Sus Scrofa",
                "common": "Wild Boar"
            }
        }

# EmailNotifications for sending email alerts
class EmailNotifications(BaseModel):
    species: str
    last_sent: datetime

    class Config:
        schema_extra = {
            "example": {
                "species": "Sus Scrofa",
                "last_sent": "2024-09-15T10:30:00Z"
            }
        }

# User Signup Schema with custom validators
class UserSignupSchema(BaseModel):
    username: str  # Username field
    password: constr(min_length=8)  # Password must be at least 8 characters long
    email: EmailStr  # Validate email format
    roles: List[str]  # List of roles assigned to the user
    gender: str  # Gender of the user
    DoB: datetime  # Date of Birth (dd/mm/yyyy)
    address: AddressSchema  # Nested address schema
    organization: str  # User's organization
    phonenumber: Optional[str]  # Optional phone number field
    notificationAnimals: Optional[List[AnimalNotificationSchema]] = []  # Animal notifications
    emailNotifications: Optional[List[EmailNotifications]] = []  # Email notifications

    # Custom date of birth validator
    @validator('DoB', pre=True)
    def validate_dob(cls, value):
        if isinstance(value, str):
            return datetime.strptime(value, '%d/%m/%Y')
        return value
    
    # Validate gender field against predefined options
    @validator('gender')
    def validate_gender(cls, value):
        if value.lower() not in GENDER:
            raise ValueError(f"{value} is not a valid gender")
        return value.upper()

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# User Login Schema for login validation
class UserLoginSchema(BaseModel):
    username: str  # Username field
    email: EmailStr  # Email must be a valid format
    password: str  # Password field

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Guest Schema for guest user information
class GuestSchema(BaseModel):
    username: str  # Guest username
    email: EmailStr  # Guest email
    password: str  # Guest password
    userId: str  # User ID for the guest
    roles: List[dict]  # Roles for the guest user
    expiresAt: datetime  # Expiry date for the guest account

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Schema for signing up a guest
class GuestSignupSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    timestamp: datetime  # Timestamp for guest account creation

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Schema for Forgot Password functionality
class ForgotPasswordSchema(BaseModel):
    user: str  # Username or email for password reset

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Schema for Resetting Password
class ResetPasswordSchema(BaseModel):
    user: str  # Username or email for password reset
    password: str  # New password
    otp: int  # OTP code for verification

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Schema for Recording Data
class RecordingData(BaseModel):
    timestamp: datetime  # Timestamp for the recording
    sensorId: str  # Sensor ID for the recording
    microphoneLLA: conlist(float, min_items=3, max_items=3)  # Microphone coordinates
    animalEstLLA: conlist(float, min_items=3, max_items=3)  # Estimated location of the animal
    animalTrueLLA: conlist(float, min_items=3, max_items=3)  # True location of the animal
    animalLLAUncertainty: float  # Uncertainty of the location
    audioClip: str  # Audio clip data (base64)
    mode: str  # Mode of the recording
    audioFile: str  # Audio file name

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
