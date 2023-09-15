import os
import pymongo

# import mongoose

# please use echonet credentials here, this connection string is just a placeholder
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_user_pswd = os.getenv("DB_USER_PASS")
db_root_user = os.getenv("DB_ROOT_USER")
db_root_user_pswd = os.getenv("DB_ROOT_USER_PASS")


connection_string=f"mongodb://{db_user}:{db_user_pswd}@{db_host}:27017/EchoNet"
#connection_string="mongodb+srv://projectechodeakin:uKRBgDwBmimUuV2Q@cluster0.gu2idc8.mongodb.net/test"
client = pymongo.MongoClient(connection_string)
db = client['EchoNet']
#db = client['mydatabase']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones



User_connection_string = f"mongodb://{db_root_user}:{db_root_user_pswd}@{db_host}/UserSample?authSource=admin"
Userclient = pymongo.MongoClient(User_connection_string)
Userdb = Userclient['UserSample']
User = Userdb.users
Role = Userdb.roles
ROLES = ["user", "admin", "guest"]