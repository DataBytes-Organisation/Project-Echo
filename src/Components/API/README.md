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
