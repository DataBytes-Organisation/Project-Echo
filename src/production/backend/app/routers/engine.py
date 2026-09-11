## app.routers.engine.py
from fastapi import status, APIRouter
from app import serializers
from app import schemas
from app.database import Events
from app.services.detection_stream import detection_stream_manager
import asyncio
import datetime
import logging
from app import serializers
from app import schemas
from app.database import Events, Species
import datetime
from fastapi.responses import StreamingResponse
import pandas as pd


logger = logging.getLogger(__name__)
router = APIRouter()


def _build_created_event(inserted_id):
    pipeline = [
        {"$match": {"_id": inserted_id}},
    ]
    return serializers.eventListEntity(Events.aggregate(pipeline))[0]


def _build_stream_payload(inserted_id):
    stream_pipeline = [
        {"$match": {"_id": inserted_id}},
        {
            "$lookup": {
                "from": "species",
                "localField": "species",
                "foreignField": "_id",
                "as": "info",
            }
        },
        {
            "$replaceRoot": {
                "newRoot": {
                    "$mergeObjects": [{"$arrayElemAt": ["$info", 0]}, "$$ROOT"]
                }
            }
        },
    ]
    return serializers.eventSpeciesListEntity(
        Events.aggregate(stream_pipeline)
    )[0]


@router.post("/event", status_code=status.HTTP_201_CREATED)
async def create_event(event: schemas.EventSchema):
    # Keep Mongo work off the event loop so open WebSocket clients stay responsive.
    result = await asyncio.to_thread(Events.insert_one, event.dict())
    # Persistence already succeeded. Broadcast failures must not turn this into a 500.
    try:
        stream_payload = await asyncio.to_thread(
            _build_stream_payload, result.inserted_id
        )
        await detection_stream_manager.broadcast(stream_payload)
    except Exception:
        logger.warning(
            "Event %s persisted but live broadcast failed",
            result.inserted_id,
            exc_info=True,
        )

    return {"status": "success", "eventId": str(result.inserted_id)}

    
# Return all species data

@router.get("/animal_records", response_description="Get all record of animals")
def list_species_data(species: str = "", event_start: str = "", event_end: str = "", microphoneLLA_0: float = None, microphoneLLA_1: float = None,  microphoneLLA_2: float = None, sourceType: str = "all"):
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
            'sourceType': "$events.sourceType",
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
    if sourceType not in ("all", "real", "simulator"):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="sourceType must be all, real, or simulator")
    if sourceType != "all":
        pipeline.append({'$match': {'sourceType': sourceType}})
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

