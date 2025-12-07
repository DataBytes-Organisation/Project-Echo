## app.database.py
import pymongo
import os
import datetime
import time
# import mongoose

# prefer environment variable inside containers; fallback to service hostname (EchoNet DB)
# legacy (kept for reference):
# connection_string = "mongodb://modelUser:EchoNetAccess2023@ts-mongodb-cont:27017/EchoNet"
connection_string = os.getenv(
    "MONGODB_URI",
    "mongodb://modelUser:EchoNetAccess2023@ts-mongodb-cont:27017/EchoNet",
)
client = pymongo.MongoClient(connection_string)
db = client["EchoNet"]
# db = client['mydatabase']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones

Nodes = db.nodes
Components = db.components
Commands = db.commands

# User DB connection (env first, then service hostname)
# legacy (kept for reference):
# User_connection_string = "mongodb://root:root_password@ts-mongodb-cont/UserSample?authSource=admin"
User_connection_string = os.getenv(
    "USER_MONGODB_URI",
    "mongodb://root:root_password@ts-mongodb-cont/UserSample?authSource=admin",
)
Userclient = pymongo.MongoClient(User_connection_string)
Userdb = Userclient['UserSample']
User = Userdb.users
Role = Userdb.roles
Guest = Userdb.guests
Requests = Userdb.requests
ForgotPassword = Userdb.forgotpasswords
LogoutToken = Userdb.logouttokens

ROLES = ["user", "admin", "guest"]
STATES_CODE = ["vic", "nsw", "ts", "ql", "sa", "wa"]
GENDER = ["male", "female", "m", "f", "prefer not to say"]
AUS_STATES = ["victoria", "newsouthwales", "tasmania", "queensland", "southaustralia", "westernaustralia"]

# # Define the interval in seconds (6 minutes)
# interval_seconds = 360

# # Check and delete expired guests data 
# while True:
#     now = datetime.datetime.now()
#     print("Background monitor at", now)

#     # Define the cutoff time for deleting records
#     cutoff_time = now - datetime.timedelta(seconds=interval_seconds)

#     # Delete expired documents
#     deleted_count = Guest.delete_many({"expiresAt": {"$lte": cutoff_time}}).deleted_count
#     print(f"Deleted {deleted_count} expired documents.")

#     # Sleep for the specified interval before running again
#     time.sleep(interval_seconds)

# Update Database Setup (t2.2025)
AudioUploads = db.audio_uploads
Predictions = db.predictions
Detections = db.detections