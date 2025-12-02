
import os
import time
import logging
import threading
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

# Routers
from .routers import add_csv_output_option, audio_upload_router
from app.routers import species_predictor, auth_router, hmi, engine, sim, two_factor, public, iot, live #Websocket

# --- FastAPI App Setup ---
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(audio_upload_router.router, tags=['audio'], prefix='/api')
app.include_router(hmi.router, tags=['hmi'], prefix='/hmi')
app.include_router(engine.router, tags=['engine'], prefix='/engine')
app.include_router(sim.router, tags=['sim'], prefix='/sim')
app.include_router(two_factor.router)
app.include_router(public.router, tags=['public'], prefix='/public')
app.include_router(iot.router, tags=['iot'], prefix='/iot')
app.include_router(species_predictor.router, tags=["predict"])
app.include_router(auth_router.router, tags=["auth"], prefix="/api")
app.include_router(live.router) #Websocket 

# --- Root Endpoint ---
@app.get("/", response_description="API Root")
def show_home():
    return 'Welcome to echo api, move to /docs for more'

# --- OpenAPI Spec Endpoints ---
@app.get("/openapi-export", include_in_schema=False)
async def get_openapi_spec():
    return app.openapi()

@app.get("/spec/summary", tags=["debug"], include_in_schema=False)
async def get_spec_summary():
    spec = app.openapi()
    return {
        "title": spec.get("info", {}).get("title"),
        "version": spec.get("info", {}).get("version"),
        "number_of_paths": len(spec.get("paths", {})),
        "tags": [tag.get("name") for tag in spec.get("tags", []) if "name" in tag]
    }

def export_openapi_to_file():
    output_dir = "backend"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "project-echo-openapi.json")
    with open(output_path, "w") as f:
        json.dump(app.openapi(), f, indent=2)
    print(f"✅ OpenAPI spec exported to {output_path}")

export_openapi_to_file()

# --- 24/7 Engine Background Task ---
def continuous_engine_task():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()]
    )
    while True:
        try:
            # Replace with your actual processing logic
            logging.info("\n [ Place holder for future engine tasks~ ]")
            # Example: engine.process_new_data()
            time.sleep(5)  # Run every 5 seconds
        except Exception as e:
            logging.error(f"Engine error: {e}")
            time.sleep(5)

def start_background_engine():
    thread = threading.Thread(target=continuous_engine_task, daemon=True)
    thread.start()
    logging.info("Continuous engine background task started.")


start_background_engine()



