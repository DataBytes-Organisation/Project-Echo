from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import bcrypt

from app.database import User

from app.middleware.auth import signJWT
from app.middleware.random import genotp
import time
import requests
import os

router = APIRouter()

otp_store = {}

# reCAPTCHA configuration
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY", "6Lee1k0sAAAAAH33-o7w2ghN5suNAD8UMkP5lOOT")
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"

class SignInRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str
    recaptchaToken: Optional[str] = None  # Optional for backward compatibility

class OTPVerifyRequest(BaseModel):
    email: str
    otp: str

def verify_recaptcha(token: str) -> bool:
    """Verify reCAPTCHA token with Google"""
    if not token:
        return False
    
    try:
        response = requests.post(
            RECAPTCHA_VERIFY_URL,
            data={
                "secret": RECAPTCHA_SECRET_KEY,
                "response": token
            }
        )
        result = response.json()
        # For reCAPTCHA v3, check the score (0.0 to 1.0)
        # Higher score = more likely legitimate, lower = more suspicious
        return result.get("success", False) and result.get("score", 0) > 0.5
    except Exception as e:
        print(f"reCAPTCHA verification error: {e}")
        return False

@router.post("/signin")
def signin(data: SignInRequest):
    # Verify reCAPTCHA token if provided
    if data.recaptchaToken:
        if not verify_recaptcha(data.recaptchaToken):
            raise HTTPException(status_code=403, detail="reCAPTCHA verification failed")
    
    # Check that at least email or username is provided
    if not data.email and not data.username:
        raise HTTPException(status_code=400, detail="Either email or username is required")
    
    # Find user by email or username
    query = {}
    if data.email:
        query["email"] = data.email
    if data.username:
        query["username"] = data.username
    
    user = User.find_one(query) if query else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password using bcrypt
    try:
        stored_password = user.get("password", "")
        # Check if password is already hashed (bcrypt hashes start with $2a$, $2b$, or $2y$)
        if stored_password.startswith(('$2a$', '$2b$', '$2y$')):
            # Password is hashed, use bcrypt
            is_valid = bcrypt.checkpw(data.password.encode('utf-8'), stored_password.encode('utf-8'))
        else:
            # Fallback to plaintext comparison (for backwards compatibility)
            is_valid = (stored_password == data.password)
        
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        print(f"Password verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    otp = genotp()
    otp_store[user.get("email", data.email or data.username)] = {
        "otp": otp,
        "timestamp": time.time(),
        "user": user
    }

    print(f"[DEBUG] OTP for {user.get('email')} is {otp}")  # Simulate email

    return {"message": "OTP sent to email", "email": user.get("email")}

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
