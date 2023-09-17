import os
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import datetime
from app import serializers
from app import schemas
import pymongo
import json
from app.routers import hmi, engine, sim
app = FastAPI()

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your own allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hmi.router, tags=['hmi'], prefix='/hmi')
app.include_router(engine.router, tags=['engine'], prefix='/engine')
app.include_router(sim.router, tags=['sim'], prefix='/sim')


# Load the project echo credentials into a dictionary

'''try:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_config.json')
    with open(file_path, 'r') as f:
        echo_config = json.load(f)
    print(f"Echo API echo_config successfully loaded", flush=True)
except:
    print(f"Could not API echo_config : {file_path}") 
print(f" database names: {client.list_database_names()}")
'''

@app.get("/", response_description="api-root")
def show_home():
    return 'Welcome to echo api, move to /docs for more'

