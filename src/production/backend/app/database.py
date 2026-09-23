## app.database.py
import logging
import os
import pymongo
import datetime
import time

from app.config import settings

logger = logging.getLogger(__name__)

MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", str(settings.mongo_timeout_ms)))
MONGODB_CLIENT_OPTIONS = {
    "serverSelectionTimeoutMS": MONGODB_TIMEOUT_MS,
    "connectTimeoutMS": MONGODB_TIMEOUT_MS,
    "socketTimeoutMS": MONGODB_TIMEOUT_MS,
}

# Primary EchoNet DB connection
client = pymongo.MongoClient(settings.mongodb_uri, **MONGODB_CLIENT_OPTIONS)
db = client[settings.mongo_db_name]

Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones
Donations = db.donations
RazorpayOrders = db.razorpay_orders

Nodes = db.nodes
Components = db.components
Commands = db.commands

# Sensor Health (Admin)
SensorSettings = db.sensor_settings
SensorReboots = db.sensor_reboots

# User DB connection
Userclient = pymongo.MongoClient(settings.user_mongodb_uri, **MONGODB_CLIENT_OPTIONS)
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

# Audio, Predictions & Detections
AudioUploads = db.audio_uploads
Predictions = db.predictions
Detections = db.detections
AdminBudgets = db.admin_budgets
ServiceStates = db.service_states
Projects = db["projects"]


def init_indexes():
    """Safely initialize database indexes without blocking module imports if DB is temporarily unreachable."""
    try:
        Events.create_index([("sourceType", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)], name="idx_events_source_type_timestamp")
        SensorSettings.create_index([("_id", pymongo.ASCENDING)], name="idx_sensor_settings_id")
        SensorReboots.create_index(
            [("sensorId", pymongo.ASCENDING), ("requestedAt", pymongo.DESCENDING)],
            name="idx_sensor_reboots_sensor_requestedAt_desc",
        )
        Detections.create_index([("species", pymongo.ASCENDING)], name="idx_species")
        Detections.create_index([("timestamp", pymongo.DESCENDING)], name="idx_timestamp_desc")
        Detections.create_index(
            [("species", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
            name="idx_species_timestamp_desc",
        )
        try:
            Detections.drop_index("idx_microphone_lat_lon")
        except Exception:
            pass
        Detections.create_index(
            [("microphoneLLA.latitude", pymongo.ASCENDING), ("microphoneLLA.longitude", pymongo.ASCENDING)],
            name="idx_microphone_lla_obj",
        )
        Projects.create_index("name")
        Projects.create_index("status")
        Projects.create_index("location")
        Projects.create_index("ecologists")
    except Exception as e:
        logger.debug("Database index creation deferred/skipped: %s", e)


init_indexes()
