import os
import json
# from Components.API.app.routers import add_csv_output_option, audio_upload_router
from .routers import add_csv_output_option, audio_upload_router

from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from app.routers import species_predictor
from app.routers import auth_router
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import datetime
import pymongo
import json 


# from .routers import hmi, engine, sim
app = FastAPI()

# Add the CORS middleware

from app.routers import hmi, engine, sim, two_factor
app = FastAPI()




from app.routers import hmi, engine, sim
from app.routers import public
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # 可根据实际需求配置


from app.routers import hmi, engine, sim, iot

# ✅ Add metadata here
app = FastAPI(
    title="Project Echo API",
    description="""
    Project Echo is an IoT-based system designed to record and analyze audio data for species identification and ecosystem monitoring.

    This API provides endpoints to:
    - Upload audio files
    - Simulate audio responses
    - Interface with HMI and audio engine modules
    """,
    version="1.0.0"
)

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.include_router(hmi.router, tags=['hmi'], prefix='/hmi')
# app.include_router(engine.router, tags=['engine'], prefix='/engine')
# app.include_router(sim.router, tags=['sim'], prefix='/sim')
# app.include_router(add_csv_output_option.router, tags=['csv'], prefix='/api')
app.include_router(audio_upload_router.router, tags=['audio'], prefix='/api')


# ✅ Include routers
app.include_router(hmi.router, tags=['hmi'], prefix='/hmi')
app.include_router(engine.router, tags=['engine'], prefix='/engine')
app.include_router(sim.router, tags=['sim'], prefix='/sim')
app.include_router(two_factor.router)


app.include_router(public.router, tags=['public'], prefix='/public')

'''try:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_config.json')
    with open(file_path, 'r') as f:
        echo_config = json.load(f)
    print(f"Echo API echo_config successfully loaded", flush=True)
except:
    print(f"Could not API echo_config : {file_path}") 
print(f" database names: {client.list_database_names()}")
'''

app.include_router(iot.router, tags=['iot'], prefix='/iot')
app.include_router(species_predictor.router, tags=["predict"])


import os  # ✅ Ensure this is already imported at the top
