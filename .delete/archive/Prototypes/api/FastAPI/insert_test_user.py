import bcrypt
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
connection_string = f"mongodb+srv://{os.getenv('MONGO_USERNAME')}:{os.getenv('MONGO_PASSWORD')}@{os.getenv('MONGO_CLUSTER')}/{os.getenv('DATABASE_NAME', 'EchoNet')}?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client[os.getenv("DATABASE_NAME", "EchoNet")]

password = "password".encode("utf-8")
hashed = bcrypt.hashpw(password, bcrypt.gensalt()).decode("utf-8")
db["users"].insert_one({"username": "testuser", "password": hashed})
print("Test user inserted.")
