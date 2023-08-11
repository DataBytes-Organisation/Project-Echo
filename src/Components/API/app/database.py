import pymongo
# import mongoose

# please use echonet credentials here, this connection string is just a placeholder
connection_string="mongodb://modelUser:EchoNetAccess2023@ts-mongodb-cont:27017/EchoNet"
#connection_string="mongodb+srv://projectechodeakin:uKRBgDwBmimUuV2Q@cluster0.gu2idc8.mongodb.net/test"
client = pymongo.MongoClient(connection_string)
db = client['EchoNet']
#db = client['mydatabase']
Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones

User_connection_string = "mongodb://root:root_password@ts-mongodb-cont/UserSample?authSource=admin"
Userdb = pymongo.MongoClient(User_connection_string)
User = Userdb.user
Userdb.role = ["user", "admin"]
Role = Userdb.role