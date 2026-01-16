## app.schemas.py
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator, constr, conlist, condecimal
from bson.objectid import ObjectId
from app.database import GENDER, STATES_CODE, AUS_STATES

# Custom class to validate MongoDB ObjectId
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):  # Check if the ObjectId is valid
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        # Update schema to represent ObjectId as a string
        field_schema.update(type="string")

# Schema to validate event data with input fields like sensorId, location, etc.
class EventSchema(BaseModel):
    timestamp: datetime  # Event timestamp
    sensorId: constr(min_length=1)  # Non-empty string for sensor ID
    species: constr(min_length=1)  # Non-empty string for species name
    microphoneLLA: conlist(float, min_items=3, max_items=3)  # List of exactly 3 floats for microphone location
    animalEstLLA: conlist(float, min_items=3, max_items=3)  # List of exactly 3 floats for estimated animal location
    animalTrueLLA: conlist(float, min_items=3, max_items=3)  # List of exactly 3 floats for true animal location
    animalLLAUncertainty: int  # Uncertainty value
    audioClip: str  # Audio clip data
    confidence: condecimal(gt=0, lt=100)  # Confidence value between 0 and 100
    sampleRate: int  # Audio sample rate

    # Configuration and schema example
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
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

# Schema to validate movement data such as animal location, timestamp, etc.
class MovementSchema(BaseModel):
    timestamp: datetime  # Movement timestamp
    species: constr(min_length=1)  # Non-empty string for species name
    animalId: constr(min_length=1)  # Non-empty string for animal ID
    animalTrueLLA: conlist(float, min_items=3, max_items=3)  # List of exactly 3 floats for true animal location

    # Configuration and schema example
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "timestamp": "2023-03-22T13:45:12.000Z",
                "species": "Sus Scrofa",
                "animalId": "1",
                "animalTrueLLA": [-33.1106, 150.0570, 23]
            }
        }

# Schema to validate microphone location data
class MicrophoneSchema(BaseModel):
    sensorId: constr(min_length=1)  # Non-empty string for sensor ID
    microphoneLLA: conlist(float, min_items=3, max_items=3)  # List of exactly 3 floats for microphone location

    # Configuration and schema example
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "sensorId": "2",
                "microphoneLLA": [-33.1106, 150.0570, 23]
            }
        }

# Schema to validate address data with custom validation for Australian states
class AddressSchema(BaseModel):
    country: str  # Country name
    state: Optional[str]  # State name (optional, but required if country is Australia)

    # Custom validator to ensure state is required and valid if country is Australia
    @validator('state', pre=True, always=True)
    def validate_state(cls, value, values):
        if 'country' in values and values['country'].lower() in ["australia", "aus"]:
            if not value:
                raise ValueError("State is required for Australia")
            if value.lower() not in STATES_CODE + AUS_STATES:
                raise ValueError(f"{value} is not a valid state for Australia")
        return value

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema to handle user request for animal data changes
class RequestSchema(BaseModel):
    username: str  # Username of the requester
    animal: str  # Animal name
    requestingToChange: str  # What field is requested to be changed
    initial: str  # Initial value
    modified: str  # Modified value
    source: str  # Source of request
    date: datetime  # Date of request
    status: str  # Request status

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema to define animal notification structure
class ComponentSchema(BaseModel):
    type: str
    model: str
    customProperties: Dict[str, str]

class LocationSchema(BaseModel):
    latitude: float
    longitude: float

class NodeSchema(BaseModel):
    _id: str
    name: str
    type: str
    model: str
    location: LocationSchema
    customProperties: Dict[str, str]
    components: List[ComponentSchema] = []
    connectedNodes: List[str] = []

class AnimalNotificationSchema(BaseModel):
    species: str
    confidence: float
    common: str  # Common name

    # Schema example for Swagger UI documentation
    class Config:
        schema_extra = {
            "example": {
                "species": "Sus scrofa",
                "common": "Wild Boar"
            }
        }

# Schema to handle email notification timing
class EmailNotifications(BaseModel):
    species: str  # Species name
    last_sent: datetime  # Last sent timestamp

    # Schema example for Swagger UI documentation
    class Config:
        schema_extra = {
            "example": {
                "species": "Sus scrofa",
                "last_sent": "2024-09-15T10:30:00Z"
            }
        }

# Schema to handle user signup with custom validations for DoB, gender, and address
class UserSignupSchema(BaseModel):
    username: str  # Username
    password: constr(min_length=8)  # Password with minimum 8 characters
    email: str  # Email address
    roles: List[str]  # List of roles
    gender: str  # Gender
    DoB: datetime  # Date of birth
    address: AddressSchema  # Address details
    organization: str  # Organization name
    phonenumber: Optional[str]  # Phone number (optional)
    notificationAnimals: Optional[List[AnimalNotificationSchema]] = []  # List of notification animals (optional)
    emailNotifications: Optional[List[EmailNotifications]] = []  # List of email notifications (optional)

    # Custom validator for date of birth format
    @validator('DoB', pre=True)
    def validate_dob(cls, value: object) -> object:
        try:
            if isinstance(value, str):
                return datetime.combine(datetime.strptime(value, '%d/%m/%Y'), datetime.min.time())
        except ValueError:
            raise ValueError('Invalid date format. Please use "dd/mm/yyyy".')

    # Custom validator for address validation, especially for Australian states
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

    # Custom validator for gender field
    @validator('gender')
    def validate_gender(cls, value):
        if value.lower() not in GENDER:
            raise ValueError('Please select the available gender.')
        return value.upper()

    # Configuration for custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema for handling user login
class UserLoginSchema(BaseModel):
    username: str  # Username
    email: str  # Email address
    password: str  # Password

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema for handling guest user details
class GuestSchema(BaseModel):
    username: str  # Guest username
    email: str  # Guest email
    password: str  # Guest password
    userId: str  # Guest user ID
    roles: List[dict]  # Guest roles
    expiresAt: datetime  # Expiration date for guest access

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema for guest signup details
class GuestSignupSchema(BaseModel):
    username: str  # Guest username
    email: str  # Guest email
    password: str  # Guest password
    timestamp: datetime  # Timestamp for signup

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema for forgot password request
class ForgotPasswordSchema(BaseModel):
    user: str  # Username for forgot password request

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema for resetting password with OTP
class ResetPasswordSchema(BaseModel):
    user: str  # Username for reset password
    password: str  # New password
    otp: int  # One-time password (OTP)

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {}

# Schema to handle recording data from the system
class RecordingData(BaseModel):
    timestamp: datetime  # Timestamp for recording
    sensorId: str  # Sensor ID
    microphoneLLA: conlist(float, min_items=3, max_items=3)  # List of 3 floats for microphone location
    animalEstLLA: conlist(float, min_items=3, max_items=3)  # List of 3 floats for estimated animal location
    animalTrueLLA: conlist(float, min_items=3, max_items=3)  # List of 3 floats for true animal location
    animalLLAUncertainty: condecimal(gt=0)  # Uncertainty greater than 0
    audioClip: str  # Audio clip data
    mode: str  # Recording mode
    audioFile: str  # Path to audio file

    # Configuration for allowing custom JSON encoders
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
           
        } 

class TwoFactorVerifySchema(BaseModel):
    user_id: str
    otp: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "user_id": "user-uuid-here",
                "otp": "123456"
            }
        }
        schema_extra = {}


class DetectionCreate(EventSchema):
    pass


class Detection(EventSchema):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "651f2a9f4d1f1b1c3e2a4567",
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

class DetectionCreate(EventSchema):
    pass

class Detection(EventSchema):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "651f2a9f4d1f1b1c3e2a4567",
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

class DetectionListResponses(BaseModel):
    items: List[Detection]
    total: int
    page: int
    page_size: int

    class config:
        allow_population_by_fiels_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Backwards/expected import name used by app.routers.detections
class DetectionListResponse(DetectionListResponses):
    pass