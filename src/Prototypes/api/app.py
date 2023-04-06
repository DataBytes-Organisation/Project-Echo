import os
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import motor.motor_asyncio
import datetime
from serializers import eventEntity, eventListEntity
import models
import pymongo

app = FastAPI()
router = APIRouter()
client = pymongo.MongoClient("mongodb://localhost:27017")
#client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client.mydatabase


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


class StudentModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(...)
    email: EmailStr = Field(...)
    course: str = Field(...)
    gpa: float = Field(..., le=4.0)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "name": "Jane Doe",
                "email": "jdoe@example.com",
                "course": "Experiments, Science, and Fashion in Nanophotonics",
                "gpa": "3.0",
            }
        }


'''
@app.post("/", response_description="Add new student", response_model=StudentModel)
async def create_student(student: StudentModel = Body(...)):
    student = jsonable_encoder(student)
    new_student = await db["students"].insert_one(student)
    created_student = await db["students"].find_one({"_id": new_student.inserted_id})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=created_student)
'''

@app.get("/event", response_description="List all events")
def list_events():
    events = eventListEntity(db["events"].find())
    print(events)
    return events


@app.get("/species", response_description="Get events with certain species")
def show_event(species: str):
    print(species)
    events = eventListEntity(db["events"].find({"species": species}))
    print(events)
    return events

@app.get("/time", response_description="Get events within certain duration")
def show_event_from_time(start: str, end: str):
    print(start, end)
    datetime_start = datetime.datetime.strptime(start, '%Y-%m-%d-%H-%M-%S')
    datetime_end = datetime.datetime.strptime(end, '%Y-%m-%d-%H-%M-%S')
    aggregate = [
        
        {
            '$match':{'timestamp': {'$lt' : datetime_end, '$gte' : datetime_start }}
        }

    ]
    events = eventListEntity(db["events"].aggregate(aggregate))
    print(events)
    return events
'''
@app.put("/{id}", response_description="Update a student", response_model=StudentModel)
async def update_student(id: str, student: UpdateStudentModel = Body(...)):
    student = {k: v for k, v in student.dict().items() if v is not None}

    if len(student) >= 1:
        update_result = await db["students"].update_one({"_id": id}, {"$set": student})

        if update_result.modified_count == 1:
            if (
                updated_student := await db["students"].find_one({"_id": id})
            ) is not None:
                return updated_student

    if (existing_student := await db["students"].find_one({"_id": id})) is not None:
        return existing_student

    raise HTTPException(status_code=404, detail=f"Student {id} not found")


@app.delete("/{id}", response_description="Delete a student")
async def delete_student(id: str):
    delete_result = await db["students"].delete_one({"_id": id})

    if delete_result.deleted_count == 1:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    raise HTTPException(status_code=404, detail=f"Student {id} not found")




     start = datetime.datetime(2023, 3, 22, 13, 44,4)
    end = datetime.datetime(2023, 3, 22, 13, 55,41)
    aggregate = [
        {
            '$match':{'timestamp': {'$lt' : end, '$gte' : start }}
        },
        {
            '$lookup': {
                'from': 'species',
                'localField': 'species',
                'foreignField': '_id',
                'as': 'info'
            }
        }
        

    ]
    #if (events := await db["events"].aggregate(aggregate)) is not None:
    #if (events := await db["events"].find({'timestamp': {'$lt' : end, '$gte' : start }}).to_list(1000)) is not None:
       # return events
    events = await db["events"].find({'species': 'Sus Scrofa'}).to_list(1000)
    return events

    #raise HTTPException(status_code=404, detail=f"Student {id} not found")
    
'''