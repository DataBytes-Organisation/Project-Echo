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
from app.database import Test_Events
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
MQTT_BROKER_PORT = 1884

@router.post("/post_points", status_code=status.HTTP_201_CREATED)
def post_recording(data: schemas.Triangulation):



    # Create the vocalisation event
    vocalisation_times = {
        #"timestamp": data.timestamp.isoformat(),
        "clusterID": data.clusterID,
        "times": data.times   
    }    
    MQTT_MSG = json.dumps(vocalisation_times)
    print("dumped")
    #print(data)

    publish.single("Triangulate", MQTT_MSG, hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)
    #publish.single("Simulate_Recording", payload= MQTT_MSG, hostname=MQTT_ENGINE_URL, port=MQTT_BROKER_PORT)

    # publish the audio message on the queue
    #(rc, mid) = mqtt_client.publish(MQTT_BROKER_URL, MQTT_MSG)
    print("sent")

    return data


@router.post("/update_record", status_code=status.HTTP_200_OK)
def test_event(event: schemas.Location):
    

    filter = {'clusterID': event.clusterID }

    # Define the update operation with multiple fields
    update = {
        '$set': {
            "animalEstLLA": event.sourceLLA,
            "animalTrueLLA": event.sourceLLA,
            # Add more fields as needed
        }
    }

    print("Filter:", filter)
    print("Update:", update)
    
    result = Test_Events.update_one(filter, update)
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    
    response = schemas.UpdateResponse(
        matched_count=result.matched_count,
        modified_count=result.modified_count,
        acknowledged=result.acknowledged
    )

    return response