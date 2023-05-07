from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from app import serializers
from app import schemas
from app.database import Movements

router = APIRouter()


@router.post("/movement", status_code=status.HTTP_201_CREATED)
def create_movement(movement: schemas.MovementSchema):

    result = Movements.insert_one(movement.dict())
    pipeline = [
            {'$match': {'_id': result.inserted_id}},
        ]
    new_post = serializers.movementListEntity(Movements.aggregate(pipeline))[0]
    return new_post