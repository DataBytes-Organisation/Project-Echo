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

from flask import Flask, jsonify, render_template
import json

app = Flask(__name__)

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Replace with your own allowed origins
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
# Route to render the submission overview page (frontend)
@app.route('/submission-overview', methods=['GET'])
def submission_overview():
    # Load the submissions data from the 'submissions.json' file
    try:
        with open('submissions.json', 'r') as file:
            submissions_data = json.load(file)
    except FileNotFoundError:
        submissions_data = []
    
    # Render the page with the submissions data
    return render_template('submissionOverview.html', submissions=submissions_data)

# API endpoint to get the list of submissions (in case it's needed)
@app.route('/api/submissions', methods=['GET'])
def get_submissions():
    try:
        with open('submissions.json', 'r') as file:
            submissions_data = json.load(file)
        return jsonify(submissions_data)
    except FileNotFoundError:
        return jsonify([])

# Route to add a new submission (just a simple POST example)
@app.route('/submit-audio', methods=['POST'])
def submit_audio():
    # This is a simple example; in a real scenario, you would get data from a form or file upload
    # For now, let's assume we are submitting a new record
    new_submission = {
        "submission_date": "2025-04-05",  # Example date, replace with actual data
        "detected_animal": "Kangaroo",  # Example detected animal, replace with actual data
        "audio_file": "audio_file_1.wav"  # Example audio file, replace with actual file name
    }

    # Load existing submissions and add the new one
    try:
        with open('submissions.json', 'r') as file:
            submissions_data = json.load(file)
    except FileNotFoundError:
        submissions_data = []

    submissions_data.append(new_submission)

    # Save the updated submissions back to the JSON file
    with open('submissions.json', 'w') as file:
        json.dump(submissions_data, file, indent=4)

    return jsonify({"message": "Audio submission successful!"}), 201

if __name__ == '__main__':
    app.run(debug=True)
