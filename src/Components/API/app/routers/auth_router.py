from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict
import time

from app.database import User
from app.middleware.auth import signJWT
from app.middleware.random import genotp

router = APIRouter()

# In-memory store for demo. Prefer Redis with TTL in production.
otp_store: Dict[str, Dict] = {}

class SignInRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)

class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., regex=r"^\d{6}$")

@router.post("/auth/signin")
def signin(data: SignInRequest):
    # Lookup user
    try:
        user = User.find_one({"email": data.email})
    except Exception:
        raise HTTPException(status_code=500, detail="Database error while fetching user")

    if not user or not user.get("password") == data.password:
        # Replace with proper hashed password check
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Rate limit OTP issuance (1 per 30s)
    now = time.time()
    record = otp_store.get(data.email)
    if record and now - record.get("timestamp", 0) < 30:
        raise HTTPException(status_code=429, detail="Please wait before requesting another OTP")

    otp = genotp()  # must return 6-digit string
    otp_store[data.email] = {"otp": otp, "timestamp": now, "attempts": 0, "user": user}

    # TODO: Send OTP via email/SMS. For now, expose only for development (remove in prod)
    return {"message": "OTP sent", "dev_otp": otp}

@router.post("/auth/verify")
def verify(data: VerifyOtpRequest):
    record = otp_store.get(data.email)
    if not record:
        raise HTTPException(status_code=404, detail="OTP not found; sign in first")

    if time.time() - record["timestamp"] > 300:
        del otp_store[data.email]
        raise HTTPException(status_code=410, detail="OTP expired")

    if record.get("attempts", 0) >= 5:
        del otp_store[data.email]
        raise HTTPException(status_code=429, detail="Too many incorrect attempts; request a new OTP")

    if record["otp"] != data.otp:
        record["attempts"] = record.get("attempts", 0) + 1
        raise HTTPException(status_code=401, detail="Invalid OTP")

    user = record["user"]
    roles = user.get("roles", ["user"])
    try:
        token = signJWT(user, roles)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to sign JWT")

    del otp_store[data.email]
    return {"message": "Login successful", "access_token": token}
