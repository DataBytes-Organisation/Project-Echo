import pymongo
# import mongoose

# please use echonet credentials here, this connection string is just a placeholder
# connection_string="mongodb://modelUser:EchoNetAccess2023@ts-mongodb-cont:27017/EchoNet"
connection_string = "mongodb://root:root_password@localhost:27017/?authMechanism=DEFAULT"
client = pymongo.MongoClient(connection_string)
db = client['EchoNet']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones



User_connection_string = "mongodb://root:root_password@ts-mongodb-cont/UserSample?authSource=admin"
Userclient = pymongo.MongoClient(User_connection_string)
Userdb = Userclient['UserSample']
User = Userdb.users
Role = Userdb.roles
ROLES = ["user", "admin", "guest"]
STATES_CODE = ["vic", "nsw", "ts", "ql", "sa", "wa"]
GENDER = ["male", "female", "m", "f" "prefer not to say"]
AUS_STATES = ["victoria", "newsouthwales", "tasmania", "queensland", "southaustralia", "westernaustralia"]