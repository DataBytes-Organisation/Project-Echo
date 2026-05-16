from pymongo import MongoClient

uri = "mongodb+srv://s221278697_db_user:Ts1145141919810@cluster0.tecxpmu.mongodb.net/Echonet"

client = MongoClient(uri)

db = client["Echonet"]

print("Connected to MongoDB Atlas!")
print("\nCollections:")
print(db.list_collection_names())

collections = [
    "species",
    "events",
    "microphones",
    "movements"
]

print("\nDocument Counts:")

for collection in collections:
    count = db[collection].count_documents({})
    print(f"{collection}: {count} documents")
    
client.close()