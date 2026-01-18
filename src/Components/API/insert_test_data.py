import pymongo
import datetime
import os

# Connect to local MongoDB
# Using the same URI as the test script
uri = "mongodb://root:root_password@localhost:27017/EchoNet?authSource=admin"
client = pymongo.MongoClient(uri)
db = client['EchoNet']

print("Connected to MongoDB.")

# 1. Define Species Data (Reference Table)
species_data = [
    {
        "_id": "Falco cenchroides",
        "commonName": "Nankeen kestrel",
        "type": "Bird",
        "status": "Least Concern",
        "diet": "Insectivorous"
    },
    {
        "_id": "Aquila audax",
        "commonName": "Wedge-tailed Eagle",
        "type": "Bird",
        "status": "Secure",
        "diet": "Carnivorous"
    },
    {
        "_id": "Phascolarctos cinereus",
        "commonName": "Koala",
        "type": "Mammal",
        "status": "Vulnerable",
        "diet": "Herbivore"
    },
    {
        "_id": "Vombatus ursinus",
        "commonName": "Common Wombat",
        "type": "Mammal",
        "status": "Least Concern",
        "diet": "Herbivore"
    }
]

# Insert or Update Species
print("Upserting Species...")
for s in species_data:
    db.species.update_one(
        {"_id": s["_id"]}, 
        {"$set": s}, 
        upsert=True
    )
print(f"Upserted {len(species_data)} species.")

# 2. Define Tracking/Movement Data
# The image timestamp 1763691070419 corresponds to roughly Nov 23, 2025.
# We will create records around this time and some more recent (Jan 2026).

timestamps = [
    1763691070419,  # The original from the image (Nov 2025)
    1767225600000,  # Jan 1, 2026
    1768608000000,  # Jan 17, 2026 (Yesterday relative to context)
    1768694400000   # Jan 18, 2026 (Today relative to context)
]

movements_data = [
    # Record from Image
    {
        "species": "Falco cenchroides",
        "animalId": "AHRMEF5FTZ6WQEVJNT5BMW",
        "timestamp": datetime.datetime.fromtimestamp(1763691070419 / 1000.0),
        "animalTrueLLA": [-38.78609473818882, 143.538155280096, 10]
    },
    # Advanced 1: Eagle soaring high
    {
        "species": "Aquila audax",
        "animalId": "EAGLE_009_X",
        "timestamp": datetime.datetime.fromtimestamp(timestamps[1] / 1000.0),
        "animalTrueLLA": [-38.791500, 143.542100, 150] # High altitude
    },
    # Advanced 2: Koala in a tree
    {
        "species": "Phascolarctos cinereus",
        "animalId": "KOALA_KV_02",
        "timestamp": datetime.datetime.fromtimestamp(timestamps[2] / 1000.0),
        "animalTrueLLA": [-38.780111, 143.530222, 15] 
    },
    # Advanced 3: Wombat on the ground (recent)
    {
        "species": "Vombatus ursinus",
        "animalId": "WOMBAT_G_99",
        "timestamp": datetime.datetime.fromtimestamp(timestamps[3] / 1000.0),
        "animalTrueLLA": [-38.785555, 143.535555, 0] # Ground level
    },
    # Advanced 4: The Kestrel moved (same animal as image, later time)
    {
        "species": "Falco cenchroides",
        "animalId": "AHRMEF5FTZ6WQEVJNT5BMW",
        "timestamp": datetime.datetime.fromtimestamp(timestamps[3] / 1000.0),
        "animalTrueLLA": [-38.786200, 143.538300, 25] # Moved slightly and higher
    }
]

print("Inserting Movements...")
# Clear existing test data to avoid duplicates if run multiple times? 
# Maybe better to just insert.
result = db.movements.insert_many(movements_data)
print(f"Inserted {len(result.inserted_ids)} movement records.")

print("Done. Data is ready for API testing.")
