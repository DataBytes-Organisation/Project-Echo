# code for 2fa code generation endpoint with dummy database

# import random
# import datetime
# import bcrypt
# from fastapi import FastAPI, APIRouter, HTTPException, status, Body

# # 1. OTP Generator Function
# def genotp(length: int = 6) -> str:
#     """Generate a numeric OTP of the specified length."""
#     digits = "0123456789"
#     return "".join(random.sample(digits, length))

# # 2. Dummy Database Setup
# class DummyCollection:
#     def insert_one(self, document):
#         print("Simulated insert into database:", document)
#         return {"inserted_id": "dummy_id"}

# # Create a dummy database dictionary with a "twofa" collection and a dummy "users" list.
# dummy_db = {
#     "twofa": DummyCollection(),
#     "users": [
#         {
#             "_id": "dummy_user_id",
#             "username": "testuser",
#             # Store a hashed version of 'password' for testing.
#             "password": bcrypt.hashpw("password".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
#         }
#     ]
# }

# def find_user(username: str):
#     """Simulate a user lookup in the dummy database."""
#     for user in dummy_db["users"]:
#         if user["username"] == username:
#             return user
#     return None

# # 3. FastAPI Router Setup
# router = APIRouter()

# @router.post("/auth/generate-2fa", status_code=status.HTTP_200_OK)
# def generate_2fa(username: str = Body(...), password: str = Body(...)):
#     """
#     Prototype endpoint to generate a one-time password (OTP)
#     after validating user credentials using a dummy database.
#     """
#     # Simulate user lookup
#     user = find_user(username)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Validate password
#     if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     # Generate OTP
#     otp = genotp()
    
#     # Create OTP record with a timestamp and expiration
#     otp_record = {
#         "userId": user["_id"],
#         "otp": otp,
#         "generated_at": datetime.datetime.utcnow(),
#         "expires_at": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
#     }
    
#     # Simulate inserting OTP record into the "twofa" collection
#     dummy_db["twofa"].insert_one(otp_record)
    
#     return {"message": "OTP generated successfully", "otp": otp}

# # 4. FastAPI Application Setup
# app = FastAPI(title="2FA Prototype (Dummy DB)")
# app.include_router(router, tags=["2FA Prototype"])

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("auth_proto:app", host="127.0.0.1", port=8000, reload=True)



import os
import random
import datetime
import bcrypt
import pymongo
from fastapi import FastAPI, APIRouter, HTTPException, status, Body
from dotenv import load_dotenv


# Load Environment Variables
load_dotenv()  

MONGO_USERNAME = os.getenv("MONGO_USERNAME") 
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")  
MONGO_CLUSTER  = os.getenv("MONGO_CLUSTER")  
DATABASE_NAME  = os.getenv("DATABASE_NAME", "EchoNet")  

# Creating the MongoDB Connection String
connection_string = (
    f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/"
    f"{DATABASE_NAME}?retryWrites=true&w=majority"
)

try:
    client = pymongo.MongoClient(connection_string)
    db = client[DATABASE_NAME]
    print("Successfully connected to MongoDB Atlas.")
except Exception as e:
    print("Error connecting to MongoDB:", e)
    raise e


# OTP Generation Function
def genotp(length: int = 6) -> str:
    """Generate a numeric OTP of the specified length."""
    digits = "0123456789"
    return "".join(random.sample(digits, length))



# User Lookup Function
def find_user(username: str):
    """Look up a user by username in the 'users' collection."""
    user = db["users"].find_one({"username": username})
    return user


# FastAPI Router Setup
router = APIRouter()

@router.post("/auth/generate-2fa", status_code=status.HTTP_200_OK)
def generate_2fa(username: str = Body(...), password: str = Body(...)):
    """

    Generate a one-time password (OTP) for 2FA after validating user credentials.
    Expects JSON body with 'username' and 'password'.
    """
    # Lookup the user
    user = find_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate the password using bcrypt
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate an OTP
    otp = genotp()
    
    # Create an OTP record with timestamps
    otp_record = {
        "userId": str(user["_id"]),
        "otp": otp,
        "generated_at": datetime.datetime.utcnow(),
        "expires_at": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    }
    
    # Insert the OTP record into the 'twofa' collection
    db["twofa"].insert_one(otp_record)
    
    return {"message": "OTP generated successfully", "otp": otp}

# FastAPI Application Setup
app = FastAPI(title="2FA Prototype with MongoDB Atlas")
app.include_router(router, tags=["2FA Prototype"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("auth_proto:app", host="127.0.0.1", port=8000, reload=True)
