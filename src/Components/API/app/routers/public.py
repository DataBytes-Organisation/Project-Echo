# app/routers/public.py

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import datetime
from bson import ObjectId

from app.database import Events
from app import serializers
from app.database import Events, AudioUploads
from app.schemas import AudioUploadSchema

router = APIRouter()

@router.get("/public-test", status_code=status.HTTP_200_OK)
def public_test():
    """
    A simple test endpoint to confirm that the public router is loaded successfully.
    """
    return {"message": "Public router is active!"}

@router.get("/filter-data", status_code=status.HTTP_200_OK)
def filter_data(
    start_time: Optional[float] = None, 
    end_time: Optional[float] = None,
    species: Optional[str] = None,
    microphone_id: Optional[str] = None
):
    """
    Provides a public, no-authentication data filtering interface.
    Optional query parameters include:
    - start_time: float, Unix timestamp (in seconds)
    - end_time: float, Unix timestamp (in seconds)
    - species: str, species name
    - microphone_id: str, sensor ID
    
    Returns data filtered based on the given criteria.
    """

    # Dynamically construct the query conditions
    query = {}

    if start_time is not None and end_time is not None:
        dt_start = datetime.datetime.fromtimestamp(start_time)
        dt_end = datetime.datetime.fromtimestamp(end_time)
        query["timestamp"] = {"$gte": dt_start, "$lte": dt_end}
    
    if species:
        # Assume the species field is a string stored directly in Events
        query["species"] = species
    
    if microphone_id:
        # Assume the corresponding field is named sensorId
        query["sensorId"] = microphone_id

    # Execute the query
    results = list(Events.find(query))

    if not results:
        return {"message": "No data found based on the given filters."}

    serialized = serializers.eventListEntity(results)
    return serialized

@router.post("/upload-metadata", response_description="Store audio upload metadata")
def store_audio_metadata(metadata: AudioUploadSchema):
    """
    Receives audio upload metadata (including file reference, 
    predicted species, timestamp, optional user ID) and saves 
    it into the AudioUploads collection.
    """

    data_to_insert = metadata.dict()

    # data_to_insert["_id"] = str(ObjectId())

    insert_result = AudioUploads.insert_one(data_to_insert)
    
    # Retrieve inserted document
    new_record = AudioUploads.find_one({"_id": insert_result.inserted_id})
    
    # Construct a response
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Audio upload metadata stored successfully",
            "data": {
                "_id": str(new_record["_id"]),
                "fileName": new_record["fileName"],
                "predictedSpecies": new_record.get("predictedSpecies", "Unknown"),
                "uploadTimestamp": new_record["uploadTimestamp"],
                "userId": new_record.get("userId")
            }
        }
    )


# -----------------------------------------------
# Endpoint to retrieve stored metadata
# -----------------------------------------------
@router.get("/upload-metadata", response_description="List all audio upload metadata")
def list_audio_metadata():
    """
    Returns all records from the AudioUploads collection.
    """
    uploads = list(AudioUploads.find({}))
    # Convert ObjectId to string
    for doc in uploads:
        doc["_id"] = str(doc["_id"])
    return uploads
