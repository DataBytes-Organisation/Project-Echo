# MongoDB with FastAPI

This is a prototype demonstrating how to connect fastAPI with the MongoDB database and make some queries.


# Install the requirements:
pip install -r requirements.txt

# Configure the location of your MongoDB database:
set the variable in app.py client = pymongo.MongoClient("mongodb://localhost:27017") or whatever other connection string you have set up.

# Start the service:
python -m uvicorn app:app --reload

Head to localhost:8000/docs to play around with the API 
