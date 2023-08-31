from fastapi import status, APIRouter
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Optional, List

from bson import ObjectId
import datetime
from app import serializers
from app import schemas
from app.database import Events, Movements, Microphones, User, Role, ROLES, GENDER, STATES_CODE, AUS_STATES
import paho.mqtt.publish as publish
import bcrypt
from flask import jsonify
import jwt
import requests
import datetime



router = APIRouter()

MQTT_BROKER_URL = "ts-mqtt-server-cont"
MQTT_BROKER_PORT = 1883


@router.get("/events_time", response_description="Get detection events within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(float(start))
    datetime_end = datetime.datetime.fromtimestamp(float(end))
    # print(f'we think query date is {datetime_start}', flush=True)
    # print(datetime_end)
    aggregate = [
        {
            '$match':{'timestamp': { '$gte' : datetime_start, '$lt' : datetime_end}}
            
        },
        {
            '$lookup': {
                'from': 'species',
                'localField': 'species',
                'foreignField': '_id',
                'as': 'info'
            }
        },
        {
            "$replaceRoot": { "newRoot": { "$mergeObjects": [ { "$arrayElemAt": [ "$info", 0 ] }, "$$ROOT" ] } }
        },
        {
            '$project': { "audioClip": 0, "sampleRate": 0}
        },
        {
            "$addFields": {
            "timestamp": { "$toLong": "$timestamp" }}
        }       

    ]
    events = serializers.eventSpeciesListEntity(Events.aggregate(aggregate))
    return events

@router.get("/audio", response_description="returns audio given ID")
def show_audio(id: str):
    aggregate = [
        {
            '$match':{'_id': ObjectId(id)}
        },
        {
            '$project': { "audioClip": 1, "sampleRate": 1}
        }
    ]
    results = list(Events.aggregate(aggregate))
    audio = serializers.audioListEntity(results)[0]
    return audio

@router.get("/movement_time", response_description="Get true animal movement data within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(float(start))
    datetime_end = datetime.datetime.fromtimestamp(float(end))
    
    print(f'movement range: {datetime_start}  {datetime_end}')
    
    aggregate = [
        {
            '$match':{'timestamp': {'$gte' : datetime_start ,'$lt' : datetime_end}}
        },
        {
            '$lookup': {
                'from': 'species',
                'localField': 'species',
                'foreignField': '_id',
                'as': 'info'
            }
        },
        {
            "$replaceRoot": { "newRoot": { "$mergeObjects": [ { "$arrayElemAt": [ "$info", 0 ] }, "$$ROOT" ] } }
        },
        { "$project": { "info": 0 } },
        {
            "$addFields":
            {
            "timestamp": { "$toLong": "$timestamp" }
            }
        }       

    ]
    events = serializers.movementSpeciesListEntity(Movements.aggregate(aggregate))
    return events

@router.get("/microphones", response_description="returns location of all microphones")
def list_microphones():
    results = Microphones.find()
    microphones = serializers.microphoneListEntity(results)
    return microphones

@router.get("/latest_movement", response_description="returns the latest simluated movement message")
def latest_movememnt():

    aggregate = [
    { "$sort" : { "timestamp" : -1 } },
    { "$limit": 1 },
    { "$project": { "timestamp": 1 } }]

    result = list(Movements.aggregate(aggregate))
    timestamp = serializers.timestampListEntity(result)[0]
    print(timestamp)
    return timestamp



@router.post("/sim_control", status_code=status.HTTP_201_CREATED)
def post_control(control: str):

    publish.single("Simulator_Controls", control, hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)

    return control
    
@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user: schemas.UserSignupSchema):  
    # Check to see if the request body has conflict with the database
    existing_user = User.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        if existing_user['username'] == user.username:
            response = {"message": "Failed! Username is already in use!"}
        elif existing_user['email'] == user.email:
            response = {"message": "Failed! Email is already in use!"}
        elif(user.roles not in ROLES):
            response = {"message": "Failed! Role does not exist!"}
            return JSONResponse(content=response, status_code=404)
        else:
            response = {"message": "Failed! Conflict occurred."}
        return JSONResponse(content=response, status_code=409)

    #Hash password using bcrypt
    password_hashed = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt(rounds=8))
    user.password = password_hashed.decode('utf-8')

    #Store user role as an id 
    user_role_id = []
    for role_name in user.roles:
        role = Role.find_one({"name": role_name})
        user_role_id.append(role["_id"])
    user.roles = user_role_id

    #Convert into dictionary and insert into the database
    user_dict = user.dict()
    user_dict["userId"] = user.username
    user_dict["__v"] = 0
    User.insert_one(user_dict)
    response = {"message": "User was registered successfully!"}
    return JSONResponse(content=response, status_code=201)


@router.post("/signin", status_code=status.HTTP_200_OK)
def signin(user: schemas.UserLoginSchema):
    #Find if the username exist in our database
    account = User.find_one({"username": user.username})
    if(account is None):
        response = {"message": "User Not Found."}
        return JSONResponse(content=response, status_code=404)
   
    #Find if the password matches
    passwordIsValid = bcrypt.checkpw(user.password.encode('utf-8'), account['password'].encode('utf-8'))
    if (passwordIsValid == False):
        response = {"message": "Invalid Password!"}
        return JSONResponse(content=response, status_code=401)

    #Create payload for our token
    payload = {
        "id": account["userId"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=86400)
    }
    
    token = jwt.encode(payload, "echo-auth-secret-key", algorithm = "HS256")

    authorities = []
    for role_id in account['roles']:
        role = Role.find_one({"_id": ObjectId(role_id)})
        if role:
            authorities.append("ROLE_" + role['name'].upper())

    requests.session.token = token
    result = {
        "id": account["_id"],
        "username": account["username"],
        "email": account["email"],
        "role" : authorities,
    }

    response = {"message": "User Login Successfully!"}
    print(result)
    return JSONResponse(content=response, status_code=200)

@router.post("/signout", status_code=status.HTTP_200_OK)
def signout():
    try:
        requests.session.clear()
        response = {"message": "You've been signed out!"}
        return JSONResponse(content=response)
    
    except Exception as err:
        return JSONResponse({"Error": str(err)})
    
@router.post("/ChangePassword", status_code=status.HTTP_200_OK)
def passwordchange(user: schemas.UserLoginSchema, newpw: str, cfm_newpw: str):
    #Find if the username exist in our database
    account = User.find_one({"username": user.username})
    if(account is None):
        response = {"message": "User Not Found."}
        return JSONResponse(content=response, status_code=401)
   
    #Find if the password matches
    passwordIsValid = bcrypt.checkpw(user.password.encode('utf-8'), account['password'].encode('utf-8'))
    if (passwordIsValid == False):
        response = {"message": "Invalid Password!"}
        return JSONResponse(content=response, status_code=404)

    if (newpw != cfm_newpw):
        response = {"message": "New Password Must Match Each Other"}
        return JSONResponse(content=response, status_code=401)
    
    newpw_hashed = bcrypt.hashpw(newpw.encode('utf-8'), bcrypt.gensalt(rounds=8))

    User.update_one(
        {"username": account['username']},
        {"$set": {"password": newpw_hashed.decode('utf-8')}}
    )

    response = {"message": "User Password Changed Sucessfully!"}
    return JSONResponse(content=response)


@router.get("/abc", status_code=status.HTTP_200_OK)
def abc():

    response = {"message": "User Password Changed Sucessfully!"}
    return JSONResponse(content=response)