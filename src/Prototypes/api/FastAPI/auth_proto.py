import random
import datetime
import bcrypt
from fastapi import FastAPI, APIRouter, HTTPException, status, Body

# 1. OTP Generator Function
def genotp(length: int = 6) -> str:
    """Generate a numeric OTP of the specified length."""
    digits = "0123456789"
    return "".join(random.sample(digits, length))

# 2. Dummy Database Setup
class DummyCollection:
    def insert_one(self, document):
        print("Simulated insert into database:", document)
        return {"inserted_id": "dummy_id"}

# Create a dummy database dictionary with a "twofa" collection and a dummy "users" list.
dummy_db = {
    "twofa": DummyCollection(),
    "users": [
        {
            "_id": "dummy_user_id",
            "username": "testuser",
            # Store a hashed version of 'password' for testing.
            "password": bcrypt.hashpw("password".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        }
    ]
}

def find_user(username: str):
    """Simulate a user lookup in the dummy database."""
    for user in dummy_db["users"]:
        if user["username"] == username:
            return user
    return None

# 3. FastAPI Router Setup
router = APIRouter()

@router.post("/auth/generate-2fa", status_code=status.HTTP_200_OK)
def generate_2fa(username: str = Body(...), password: str = Body(...)):
    """
    Prototype endpoint to generate a one-time password (OTP)
    after validating user credentials using a dummy database.
    """
    # Simulate user lookup
    user = find_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate OTP
    otp = genotp()
    
    # Create OTP record with a timestamp and expiration
    otp_record = {
        "userId": user["_id"],
        "otp": otp,
        "generated_at": datetime.datetime.utcnow(),
        "expires_at": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    }
    
    # Simulate inserting OTP record into the "twofa" collection
    dummy_db["twofa"].insert_one(otp_record)
    
    return {"message": "OTP generated successfully", "otp": otp}

# 4. FastAPI Application Setup
app = FastAPI(title="2FA Prototype (Dummy DB)")
app.include_router(router, tags=["2FA Prototype"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("auth_proto:app", host="127.0.0.1", port=8000, reload=True)



