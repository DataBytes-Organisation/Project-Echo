import pymongo
# import mongoose
import os

# please use echonet credentials here, this connection string is just a placeholder
connection_string=f"mongodb://modelUser:EchoNetAccess2023@{os.getenv('DB_HOST')}:27017/EchoNet"
#connection_string="mongodb+srv://projectechodeakin:uKRBgDwBmimUuV2Q@cluster0.gu2idc8.mongodb.net/test"
client = pymongo.MongoClient(connection_string)
db = client['EchoNet']
#db = client['mydatabase']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones

User_connection_string = f"mongodb://root:root_password@{os.getenv('DB_HOST')}/UserSample?authSource=admin"
Userclient = pymongo.MongoClient(User_connection_string)
Userdb = Userclient['UserSample']
User = Userdb.users
Role = Userdb.roles
Requests = Userdb.requests
ROLES = ["user", "admin", "guest"]
STATES_CODE = ["vic", "nsw", "ts", "ql", "sa", "wa"]
GENDER = ["male", "female", "m", "f" "prefer not to say"]
AUS_STATES = ["victoria", "newsouthwales", "tasmania", "queensland", "southaustralia", "westernaustralia"]

print("WTF")