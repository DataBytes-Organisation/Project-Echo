# app/routers/public.py

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import datetime
from bson import ObjectId

from app.database import Events
from app import serializers

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
