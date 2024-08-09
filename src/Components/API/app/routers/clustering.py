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
MQTT_BROKER_PORT = 1884

@router.post("/post_recording", status_code=status.HTTP_201_CREATED)
def post_recording(data: schemas.RecordingDataClustering):

    # Create the vocalisation event
    vocalisation_event = {
        #"timestamp": data.timestamp.isoformat(),
        "type": data.type,
        "clusterID":  data.clusterID,
        "audioClip" : data.audioClip, 
        "audioFile" : data.audioFile      
    }    
    MQTT_MSG = json.dumps(vocalisation_event)
    print("dumped")
    #print(data)

    publish.single("Clustering", MQTT_MSG, hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)
    #publish.single("Simulate_Recording", payload= MQTT_MSG, hostname=MQTT_ENGINE_URL, port=MQTT_BROKER_PORT)

    # publish the audio message on the queue
    #(rc, mid) = mqtt_client.publish(MQTT_BROKER_URL, MQTT_MSG)
    print("sent")

    return data