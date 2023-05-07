import pymongo

# please use echonet stuff here, this connection string is just a placeholder
connection_string="mongodb+srv://projectechodeakin:uKRBgDwBmimUuV2Q@cluster0.gu2idc8.mongodb.net/test"
client = pymongo.MongoClient(connection_string)
db = client['mydatabase']


Events = db.events
Movements = db.movements
Species = db.species
Microphones = db.microphones