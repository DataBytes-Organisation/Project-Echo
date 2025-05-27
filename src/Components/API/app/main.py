import os
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import datetime
import pymongo
import json

from app.routers import hmi, engine, sim, nodes, microphones, species_predictor
from app import serializers
from app import schemas

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

# ✅ Include routers
app.include_router(hmi.router, tags=['hmi'], prefix='/hmi')
app.include_router(engine.router, tags=['engine'], prefix='/engine')
app.include_router(sim.router, tags=['sim'], prefix='/sim')
app.include_router(nodes.router, tags=['nodes'], prefix='/nodes')
app.include_router(microphones.router, tags=['microphones'], prefix='/microphones')
app.include_router(species_predictor.router, tags=["predict"])

# ✅ Root endpoint
@app.get("/", response_description="API Root")
def show_home():
    return 'Welcome to Project Echo API. Visit /docs for interactive documentation.'
