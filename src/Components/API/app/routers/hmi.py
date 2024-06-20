from fastapi import status, APIRouter
from fastapi import FastAPI, Body, HTTPException, status, APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Optional, List
import re
from bson import ObjectId
import datetime
from app import serializers
from app import schemas
from app.database import Events, Movements, Microphones, User, Role, ROLES, Requests, Guest, ForgotPassword, LogoutToken, Species
import paho.mqtt.publish as publish
import bcrypt
from flask import jsonify
import jwt
import requests
import datetime
import json
import paho.mqtt.client as paho
from app.middleware.auth import signJWT, decodeJWT
from app.middleware.auth_bearer import JWTBearer
from app.middleware.random import randompassword
from bson.objectid import ObjectId
import uuid

jwtBearer = JWTBearer()

router = APIRouter()

MQTT_BROKER_URL = "ts-mqtt-server-cont"
MQTT_ENGINE_URL = "projectecho/engine/2"
MQTT_BROKER_PORT = 1883
#mqtt_client = paho.Client()
#mqtt_client.connect(MQTT_BROKER_URL, MQTT_BROKER_PORT)


@router.get("/events_time", response_description="Get detection events within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(float(start))
    datetime_end = datetime.datetime.fromtimestamp(float(end))
    # print(f'we think query date is {datetime_start}', flush=True)
    # print(datetime_end)
    aggregate = [
        {
            '$match':{'timestamp': { '$gte' : datetime_start, '$lt' : datetime_end},
                      'animalTrueLLA': {'$ne': None}}
            
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
            '$match': {
                'info': {'$ne': []}
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


@router.post("/api/submit", dependencies=[Depends(jwtBearer)], status_code=status.HTTP_201_CREATED)
def submit_request(request: schemas.RequestSchema):
    try:
        request_dict = request.dict()
        Requests.insert_one(request_dict)
        response = {"message": "Request submitted succesfully."}
        return JSONResponse(content=response, status_code=201)
    except Exception as e:
        response = {"message": "Exception error", "exception": str(e)}
        return JSONResponse(content=response, status_code=422)

@router.post("/sim_control", status_code=status.HTTP_201_CREATED)
def post_control(control: str):

    publish.single("Simulator_Controls", control, hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)

    return control

@router.post("/post_recording", status_code=status.HTTP_201_CREATED)
def post_recording(data: schemas.RecordingData):

    # Create the vocalisation event
    vocalisation_event = {
        "timestamp": data.timestamp.isoformat(),
        "sensorId": data.sensorId,
        "microphoneLLA": data.microphoneLLA,
        "animalEstLLA": data.animalEstLLA,
        "animalTrueLLA": data.animalTrueLLA,
        "animalLLAUncertainty": 50.0,
        "audioClip" : data.audioClip, 
        "audioFile" : data.audioFile      
    }    
    MQTT_MSG = json.dumps(vocalisation_event)
    print("dumped")
    #print(data)
    publish.single("Simulate_Recording", MQTT_MSG, hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)
    #publish.single("Simulate_Recording", payload= MQTT_MSG, hostname=MQTT_ENGINE_URL, port=MQTT_BROKER_PORT)

    # publish the audio message on the queue
    #(rc, mid) = mqtt_client.publish(MQTT_BROKER_URL, MQTT_MSG)
    print("sent")

    return data
    
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
        user_role_id.append({"_id": role["_id"]})
    user.roles = user_role_id

    #Convert into dictionary and insert into the database
    user_dict = user.dict()
    user_dict["_id"] = str(uuid.uuid4())
    user_dict["userId"] = user.username
    user_dict["__v"] = 0
    User.insert_one(user_dict)
    response = {"message": "User was registered successfully!"}
    return JSONResponse(content=response, status_code=201)

@router.post("/signin", status_code=status.HTTP_200_OK)
def signin(user: schemas.UserLoginSchema):
    #Find if the username exist in our database
    account = User.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if(account is None):
        #User Not exist - Checking inside Guest Collections
        guestAcc = Guest.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
        if(guestAcc is None):
            response = {"message": "User Not Found in Database."}
            return JSONResponse(content=response, status_code=404)
        #Verify credentials in User Collections
        passwordIsValid = bcrypt.checkpw(user.password.encode('utf-8'), guestAcc['password'].encode('utf-8'))
        if (passwordIsValid == False):
            response = {"message": "Invalid Password!"}
            return JSONResponse(content=response, status_code=401)
        authorities = []
        for role_id in guestAcc['roles']:
            role = Role.find_one({"_id": role_id["_id"]})
            if role:
                authorities.append("ROLE_" + role['name'].upper())
        #Create JWT token using user info
        jwtToken = signJWT(user=guestAcc, authorities=authorities)
        
        #Assign the session token with JWT
        requests.session.token = jwtToken
        result = {
            "id": guestAcc["_id"],
            "username": guestAcc["username"],
            "email": guestAcc["email"],
            "role" : authorities,
        }

        #Set up response (FOR TESTING ONLY)
        response = {"message": "User Login Successfully!", "tkn" : jwtToken, "roles": authorities}

        #Log result
        print(result)
        return JSONResponse(content=response, status_code=200)
    
    #User Exist
    #Verify credentials in User Collections
    #Find if the password matches
    passwordIsValid = bcrypt.checkpw(user.password.encode('utf-8'), account['password'].encode('utf-8'))
    if (passwordIsValid == False):
        response = {"message": "Invalid Password!"}
        return JSONResponse(content=response, status_code=401)
    authorities = []
    for role_id in account['roles']:
        role = Role.find_one({"_id": str(role_id["_id"])})
        if role:
            authorities.append("ROLE_" + role['name'].upper())
    
    #Create JWT token using user info
    jwtToken = signJWT(user=account, authorities=authorities)
    
    #Get info for users profile:
    #Assign the session token with JWT
    requests.session.token = jwtToken
    result = {
        "username": account["username"],
        "email": account["email"],
        "role" : authorities,
    }

    #Set up response (FOR TESTING ONLY)
    response = {"message": "User Login Successfully!", "tkn" : jwtToken, "roles": authorities, 'user': result}

    #Log result
    print(result)
    return JSONResponse(content=response, status_code=200)

# Signout functionalities for API
# However developing in the Frontend can clears the token better than storing it in a Database
@router.post("/signout", status_code=status.HTTP_200_OK)
def signout(jwtToken: str):
    try:
        if jwtToken is None:
            return jsonify({'message': 'Token is missing'}), 400

        # Check if the token has been revoked
        if LogoutToken.find_one({'jwtToken' : jwtToken}):
            return jsonify({'message': 'Token has already been revoked'}), 400

        # Add the token to the revoked tokens list
        token = {}
        token['jwtToken'] = jwtToken
        LogoutToken.insert_one(token)
        response = {"message": "You've been signed out!"}
        return JSONResponse(content=response)
    
    except Exception as err:
        return JSONResponse({"Error": str(err)})

@router.post("/guestsignup", status_code=status.HTTP_200_OK)
def guestsignup(guestSign: schemas.GuestSignupSchema):
    try:
        #Hash password using bcrypt
        password_hashed = bcrypt.hashpw(guestSign.password.encode('utf-8'), bcrypt.gensalt(rounds=8))
        recorded_password = password_hashed.decode('utf-8')

        #Create a GuestShema dict and insert values accordingly
        
        role = Role.find_one({'name': 'guest'})
        role.pop("name", None) 
        guest = dict(
            username= guestSign.username,
            email= guestSign.email,
            password= recorded_password,
            userId= guestSign.username + str(uuid.uuid4()).replace("-", "")[:8],
            roles= [role],
            expiresAt= guestSign.timestamp
        )

        Guest.insert_one(guest)
        response = {"message": "Guests was registered successfully!"}
        return JSONResponse(content=response, status_code=201)

    except Exception as e:
        response = {"message": "Error creating a new Guest account", "error":  str(e)}
        return JSONResponse(content=response, status_code=500)

@router.patch('/api/requests', dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
def update_request_status(request_dict : dict):
    print("Admin Request Status Data: {}".format(request_dict))

    if request_dict["newStatus"] is None:
        response = {"message": "Invalid request status data"}
        return JSONResponse(content=response, status_code=400)
    if request_dict["requestId"] is None:
        response = {"message": "Invalid requestId data"}
        return JSONResponse(content=response, status_code=400)

    updated_request = Requests.find_one_and_update(
        {'_id': ObjectId(request_dict["requestId"])},
        {'$set': {'status': request_dict["newStatus"]}},
        return_document=True
    )

    if updated_request is None:
        response = {"message": "Request data not found"}
        return JSONResponse(content=response, status_code=404)

    response = {"message": "Update request data successful"}
    return JSONResponse(content=response, status_code=200)

@router.patch('/api/updateConservationStatus', dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
def update_request_status(request_dict : dict):
    print("AnimalStatusUpdateData : {}".format(request_dict))
    print("Animal requestAnimal value: {}".format(request_dict["requestAnimal"]))
    print("Animal newStatus value: {}".format(request_dict["newStatus"]))
    if request_dict["newStatus"] is None:
        response = {"message": "Invalid request status data"}
        return JSONResponse(content=response, status_code=400)
    if request_dict["requestAnimal"] is None:
        response = {"message": "Invalid Animal data"}
        return JSONResponse(content=response, status_code=400)

    updated_request = Species.find_one_and_update(
        {'_id': {"$regex": re.compile(re.escape(request_dict["requestAnimal"]), re.IGNORECASE)}},
        # {'_id': "Aegotheles Cristatus"},
        {'$set': {'status': request_dict["newStatus"]}},
        return_document=True
    )

    if updated_request is None:
        response = {"message": "Animal data not found"}
        return JSONResponse(content=response, status_code=404)

    response = {"message": "Update species data successful"}
    return JSONResponse(content=response, status_code=200)


@router.post("/ChangePassword", dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
def passwordchange(oldpw: str, newpw: str, cfm_newpw: str):
    #Check if user has logged out
    # if LogoutToken.find_one({'jwtToken' : jwtToken}):
    #     response = {"message": "Token does not exist"}
    #     return JSONResponse(content=response, status_code=403)   
    
    #Retrieve Token
    JWTPayload = jwtBearer.decodedUser

    #Find if the username exist in our database
    account = User.find_one({"_id": ObjectId(JWTPayload['id'])})
    if(account is None):
        response = {"message": "User Not Found."}
        return JSONResponse(content=response, status_code=404)
   
    #Find if the password matches
    passwordIsValid = bcrypt.checkpw(oldpw.encode('utf-8'), account['password'].encode('utf-8'))
    if (passwordIsValid == False):
        response = {"message": "Invalid Password!"}
        return JSONResponse(content=response, status_code=401)

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

@router.post("/admin-dashboard", dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
async def checkAdmin(user: schemas.UserLoginSchema) -> dict:
    
    try:
        isAdmin = jwtBearer.verify_role(role="admin")
        if isAdmin:
            return {"result": "User is indeed an admin"}
        else:
            return {"result": "User is not admin"}
    except Exception as e:
        return {"error" : "Something unexpected occured - {}".format(e)}
    
@router.get("/requests", dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
async def getRequests():
    try:
        isAdmin = jwtBearer.verify_role(role="admin")
        if isAdmin:
            results = Requests.find()
            return serializers.requestListEntity(results)
        else:
            return {"result": "User is not admin"}
    except Exception as e:
        return {"error" : "Something unexpected occured - {}".format(e)}

@router.post("/change-credential", dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
def changeCredential(field:str, new: str):
    #Check if user has logged out
    # if LogoutToken.find_one({'jwtToken' : jwtToken}):
    #     response = {"message": "Token does not exist"}
    #     return JSONResponse(content=response, status_code=403)   
        
    #Retrieve Payload from Token
    JWTpayload = jwtBearer.decodedUser

    #Check if fields can be changed
    _field = ['gender', 'country', 'state', 'phonenumber']
    if(field not in _field):
        response = {"message": "This field can not be changed"}
        return JSONResponse(content=response, status_code=422)
    
    if field in ['country', "state"]:
        User.update_one(
        {"_id": ObjectId(JWTpayload['id'])},
        {"$set": {"address."+field: new}}
    )
    else:
        User.update_one(
            {"_id": ObjectId(JWTpayload['id'])},
            {"$set": {field: new}}
        )

    response = {"message": f"User {field} changed sucessfully!"}
    return JSONResponse(content=response)

@router.delete("/delete-account", dependencies=[Depends(jwtBearer)], status_code=status.HTTP_200_OK)
def deleteaccount():
    #Check if user has logged out
    # if LogoutToken.find_one({'jwtToken' : jwtToken}):
    #     response = {"message": "Token does not exist"}
    #     return JSONResponse(content=response, status_code=403)   
        
    #Retrieve Token
    JWTpayload = jwtBearer.decodedUser

    #Delete an account from User
    User.delete_one({"_id": ObjectId(JWTpayload['id'])})

    response = {"message": "User has been deleted!"}
    return JSONResponse(content=response)

@router.post("/forgot-password", status_code=status.HTTP_201_CREATED)
def forgotpassword(user: schemas.ForgotPasswordSchema):
    existing_user = User.find_one({"$or": [{"username": user.user}, {"email": user.user}]})
    if existing_user:
        newpassword = randompassword()
        print(newpassword)
        password_hashed = bcrypt.hashpw(newpassword.encode('utf-8'), bcrypt.gensalt(rounds=8))

        user_dict = {}
        user_dict["userId"] = str(existing_user['_id'])
        user_dict["newPassword"] = password_hashed.decode('utf-8')
        user_dict["modified_date"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        ForgotPassword.insert_one(user_dict)

        User.update_one(
           {"$or": [{"username": user.user}, {"email": user.user}]},
           {"$set": {"password": password_hashed.decode('utf-8')}}
        )
        
        response = {"message": "Password has been reset. Please check your email", "email": existing_user["email"], "password": newpassword}
        return JSONResponse(content=response, status_code = 201)
    
    response = {"message": "User not Found!"}
    return JSONResponse(content=response, status_code = 404)


# Get request for filter algorithm phase 1

@router.get("/filter_algorithm/{algorithm_type}", status_code=status.HTTP_200_OK, response_description="returns the running filter algorithm name")
def filter_name(algorithm_type : str):
    try:
        algorithms_running = requests.get("http://ts-api-cont:9000/engine/algorithms_data")
        algorithm_name = algorithms_running.json()[algorithm_type]
        if(algorithm_name is not None):
            response = {"message": algorithm_name}
            return JSONResponse(content=response, status_code=200) 
    except Exception as e:
        return JSONResponse({"error" : "Algorithm not Found - {}".format(e)}, status_code=404)
        
    