## app.database.py
import pymongo
import datetime
import time
# import mongoose

# please use echonet credentials here, this connection string is just a placeholder
connection_string="mongodb://modelUser:EchoNetAccess2023@ts-mongodb-cont:27017/EchoNet"
# connection_string = "mongodb://root:root_password@localhost:27017/?authMechanism=DEFAULT"
#connection_string="mongodb+srv://projectechodeakin:uKRBgDwBmimUuV2Q@cluster0.gu2idc8.mongodb.net/test"
client = pymongo.MongoClient(connection_string)
db = client['EchoNet']
#db = client['mydatabase']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones

User_connection_string = "mongodb://root:root_password@ts-mongodb-cont/UserSample?authSource=admin"
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
GENDER = ["male", "female", "m", "f" "prefer not to say"]
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