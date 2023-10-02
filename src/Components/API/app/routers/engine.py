from fastapi import status, APIRouter
from app import serializers
from app import schemas
from app.database import Events
import datetime
from app import serializers
from app import schemas
from app.database import Events, Species
import datetime
from fastapi.responses import StreamingResponse
import pandas as pd


router = APIRouter()

@router.post("/event", status_code=status.HTTP_201_CREATED)
def create_event(event: schemas.EventSchema):

    result = Events.insert_one(event.dict())
    pipeline = [
            {'$match': {'_id': result.inserted_id}},
        ]
    new_post = serializers.eventListEntity(Events.aggregate(pipeline))[0]
    return new_post

    
# Return all species data

@router.get("/animal_records", response_description="Get all record of animals")
def list_species_data(species: str = "", event_start: str = "", event_end: str = "", microphoneLLA_0: float = None, microphoneLLA_1: float = None,  microphoneLLA_2: float = None):
    pipeline = [

        {'$lookup': {
            'from': 'events',
            'localField': '_id',
            'foreignField': 'species',
            'as': 'events'
        }},
        {'$unwind': {
            'path': '$events',
            'preserveNullAndEmptyArrays': True
        }},

        {'$addFields': {
            'timestamp': "$events.timestamp",
            'sensorId': "$events.sensorId",
            'microphoneLLA': "$events.microphoneLLA",
            'animalEstLLA': "$events.animalEstLLA",
            'animalTrueLLA': "$events.animalTrueLLA",
            'animalLLAUncertainty': "$events.animalLLAUncertainty",
            'audioClip': "$events.audioClip",
            'confidence': "$events.confidence",
            'sampleRate': "$events.sampleRate"
        }}
    ]

    if species:
        pipeline.append({'$match': {'_id': species}})
    if event_end and event_start:
        datetime_start = datetime.datetime.fromtimestamp(float(event_start))
        datetime_end = datetime.datetime.fromtimestamp(float(event_end))
        pipeline.append(
            {'$match': {'timestamp': {'$gte': datetime_start, '$lt': datetime_end}}})
    if (microphoneLLA_0):
        pipeline.append({'$match': {'microphoneLLA.0': microphoneLLA_0}})
    if (microphoneLLA_1):
        pipeline.append({'$match': {'microphoneLLA.1': microphoneLLA_1}})
    if (microphoneLLA_2):
        pipeline.append({'$match': {'microphoneLLA.2': microphoneLLA_2}})

    # Convering to csv format
    df = pd.DataFrame(serializers.animalListEntity(
        list(Species.aggregate(pipeline))))
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data.csv"})


@router.get("/algorithms_data", response_description="returns the running list of algorithm data")
def filter_name():
    algorithm_name = {'Echo-Engine' : "Echo-Engine-Algorithm", 'Echo-Simulator' : "Echo-Simulator-Algorithm", 
                      'Echo-search' : "Echo-Search-Algorithm", 'Echo-lookup' : "Echo-lookup-Algorithm" }
    return algorithm_name


