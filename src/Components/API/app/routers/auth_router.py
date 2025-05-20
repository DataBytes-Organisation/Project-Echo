from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from Components.API.app.database import User
# from ..db import User
from ..middleware.auth import signJWT
from ..middleware.random import genotp
import time

router = APIRouter()

otp_store = {}

class SignInRequest(BaseModel):
    email: str
    password: str

class OTPVerifyRequest(BaseModel):
    email: str
    otp: str

@router.post("/signin")
def signin(data: SignInRequest):
    user = User.find_one({"email": data.email})
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    otp = genotp()
    otp_store[data.email] = {
        "otp": otp,
        "timestamp": time.time(),
        "user": user
    }

    print(f"[DEBUG] OTP for {data.email} is {otp}")  # Simulate email

    return {"message": "OTP sent to email"}

@router.post("/verify-otp")
def verify_otp(data: OTPVerifyRequest):
    record = otp_store.get(data.email)
    if not record or record["otp"] != data.otp:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")

    if time.time() - record["timestamp"] > 300:
        raise HTTPException(status_code=410, detail="OTP expired")

    user = record["user"]
    roles = user.get("roles", ["user"])  # fallback to "user"
    token = signJWT(user, roles)

    del otp_store[data.email]

    return {"message": "Login successful", "access_token": token}
