## app.database.py
import pymongo
import datetime
import time
# import mongoose

from app.config import settings

client = pymongo.MongoClient(settings.mongodb_uri)
db = client[settings.mongo_db_name]
# db = client['mydatabase']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones

Nodes = db.nodes
Components = db.components
Commands = db.commands

# Sensor Health (Admin)
SensorSettings = db.sensor_settings
SensorReboots = db.sensor_reboots

SensorSettings.create_index([("_id", pymongo.ASCENDING)], name="idx_sensor_settings_id")
SensorReboots.create_index(
    [("sensorId", pymongo.ASCENDING), ("requestedAt", pymongo.DESCENDING)],
    name="idx_sensor_reboots_sensor_requestedAt_desc",
)

# User DB connection
Userclient = pymongo.MongoClient(settings.user_mongodb_uri)
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

Detections.create_index([("species", pymongo.ASCENDING)], name = "idx_species")
Detections.create_index([("timestamp", pymongo.DESCENDING)], name = "idx_timestamp_desc")
Detections.create_index(
    [("species", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
    name="idx_species_timestamp_desc"
)
Detections.create_index(
    [("microphoneLLA.0", pymongo.ASCENDING), ("microphoneLLA.1", pymongo.ASCENDING)],
    name="idx_microphone_lat_lon"
)

AdminBudgets = db.admin_budgets
ServiceStates = db.service_states

Projects = db["projects"]
Projects.create_index("name")
Projects.create_index("status")
Projects.create_index("location")
Projects.create_index("ecologists")