from fastapi import Body, HTTPException, status, APIRouter
from app import serializers
from app import schemas
from app.database import Events


router = APIRouter()

@router.post("/event", status_code=status.HTTP_201_CREATED)
def create_event(event: schemas.EventSchema):

    result = Events.insert_one(event.dict())
    pipeline = [
            {'$match': {'_id': result.inserted_id}},
        ]
    new_post = serializers.eventListEntity(Events.aggregate(pipeline))[0]
    return new_post

