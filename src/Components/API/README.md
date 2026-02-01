# MongoDB with FastAPI

This is a prototype demonstrating how to connect fastAPI with the MongoDB database and make some queries.


# Install the requirements:
pip install -r requirements.txt

# Configure the location of your MongoDB database:
set the variable in app.py client = pymongo.MongoClient("mongodb://localhost:27017") or whatever other connection string you have set up.

# Start the service:

open conda shell and cd to the api folder, then run:

python -m uvicorn app.main:app --reload

Head to localhost:9000/docs to read the documentation for API 




# TEAM PROJECT T3 11/2025 NOTES NGUYEN GIA KHANG TRIEU
# If you're interested in any of these features,use the link bellow. 
# https://docs.google.com/document/d/1XtzbgMz1Yt6OSmrCM1hzhCpC7rgfgT1oK9T17tzbUnc/edit?usp=sharing

# 1. Update Engine to run 24/7 with continuous processing
# 2. WebSocket for real-time detections

# https://docs.google.com/document/d/1VxjDyqtyD9dx48-H1Za72js-g5GKcAFMDSzD3KdIQa0/edit?usp=sharing 

# Backend API for Sensor Health 

