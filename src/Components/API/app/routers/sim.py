from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List
from app import serializers
from app import schemas
from app.database import Movements, Microphones

router = APIRouter()


@router.post("/movement", status_code=status.HTTP_201_CREATED)
def create_movement(movement: schemas.MovementSchema):

    result = Movements.insert_one(movement.dict())
    pipeline = [
            {'$match': {'_id': result.inserted_id}},
        ]
    new_post = serializers.movementListEntity(Movements.aggregate(pipeline))[0]
    return new_post

@router.post("/microphones", status_code=status.HTTP_201_CREATED)
def create_microphones(microphones: List[schemas.MicrophoneSchema]):
    Microphones.drop()
    microphone_list = []
    for microphone in microphones:
        print(microphone)

        Microphones.insert_one(microphone.dict())
        microphone_list.append(microphone.dict())
        
    return JSONResponse(content = microphone_list)