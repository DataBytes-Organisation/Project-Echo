import pymongo

# please use echonet credentials here, this connection string is just a placeholder
#connection_string="mongodb+srv://projectechodeakin:uKRBgDwBmimUuV2Q@cluster0.gu2idc8.mongodb.net/test"
connection_string="mongodb://modelUser:EchoNetAccess2023@ts-mongodb-cont:27017/EchoNet"
client = pymongo.MongoClient(connection_string)
db = client['EchoNet']

Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones