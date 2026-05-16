from fastapi import APIRouter, HTTPException, Depends, status
from ..database import User, Userdb, Role
from ..middleware.auth import decodeJWT, signJWT
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from .. import schemas
from ..utils.sms import send_sms
import random
import datetime

router = APIRouter(
    tags=["2FA Authentication"]
)

security = HTTPBearer()

def generate_otp():
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))

@router.post("/2fa/generate")
async def generate_2fa_code(credentials = Depends(security)):
    """
    Generate a 2FA code for a validated user and send it via SMS
    """
    token = credentials.credentials
    # Decode JWT to get user info
    payload = decodeJWT(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get("id")
    
    # Get user details
    user = User.find_one({"_id": user_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user has phone number
    phone_number = user.get("phonenumber")
    if not phone_number:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not have a phone number registered"
        )
    
    # Generate OTP
    otp = generate_otp()
    
    # Store OTP with 5-minute expiration
    expiration = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    
    # Update user document with OTP info
    update_result = User.update_one(
        {"_id": user_id},
        {
            "$set": {
                "two_factor": {
                    "code": otp,
                    "expires_at": expiration
                }
            }
        }
    )
    
    if update_result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Failed to update user with OTP"
        )
    
    # Send OTP via SMS
    message = f"Your Project Echo verification code is: {otp}. Valid for 5 minutes."
    
    if not phone_number.startswith('+'):
        phone_number = '+61' + phone_number[1:] if phone_number.startswith('0') else '+61' + phone_number
    
    sms_sent = send_sms(phone_number, message)
    if not sms_sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP via SMS"
        )
    
    return {
        "message": "2FA code sent successfully via SMS",
        "expires_at": expiration
    }

@router.post("/2fa/verify")
async def verify_2fa_code(verify_data: schemas.TwoFactorVerifySchema):
    """
    Verify a 2FA code and issue a JWT token if valid
    """
    # Find the user
    user = User.find_one({"_id": verify_data.user_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user has a valid 2FA code
    two_factor = user.get("two_factor", {})
    if not two_factor:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No 2FA code has been generated for this user"
        )
    
    # Check if code has expired
    expiry = two_factor.get("expires_at")
    if not expiry or datetime.datetime.utcnow() > expiry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA code has expired"
        )
    
    # Verify the code
    if verify_data.otp != two_factor.get("code"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid 2FA code"
        )
    
    # Clear the 2FA code after successful verification
    User.update_one(
        {"_id": verify_data.user_id},
        {"$unset": {"two_factor": ""}}
    )
    
    # Get user roles for JWT
    authorities = []
    for role_id in user.get('roles', []):
        role = Role.find_one({"_id": str(role_id["_id"])})
        if role:
            authorities.append("ROLE_" + role['name'].upper())
    result = {
        "username": user["username"],
        "email": user["email"],
        "role" : authorities,
    }
    # Generate JWT token
    jwt_token = signJWT(user=user, authorities=authorities)
    
    return {
        "message": "2FA verification successful",
        "tkn": jwt_token,
        "roles": authorities,
        "user": result
    }
