import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException
from pymongo import MongoClient
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

router = APIRouter()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["echo"]
users_collection = db["users"]

# Mailtrap SMTP Config
conf = ConnectionConfig(
    MAIL_USERNAME="a9aa3f3c4238a8",
    MAIL_PASSWORD="9e0c4d0323682f",
    MAIL_FROM="no-reply@example.com",
    MAIL_PORT=587,
    MAIL_SERVER="sandbox.smtp.mailtrap.io",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

# Pydantic schema
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

# Registration Endpoint
@router.post("/register")
async def register_user(user: UserRegister):
    # Check if email already exists
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Generate token + expiry
    token = str(uuid.uuid4())
    expiry = datetime.utcnow() + timedelta(hours=1)

    # Insert user with verification token
    users_collection.insert_one({
        "username": user.username,
        "email": user.email,
        "password": user.password,  # ⚠️ Consider hashing this!
        "isVerified": False,
        "verificationToken": token,
        "tokenExpiry": expiry
    })

    # Email body + link
    verification_url = f"http://localhost:8000/auth/verify/{token}"
    message = MessageSchema(
        subject="Verify Your Email",
        recipients=[user.email],
        body=f"<p>Click <a href='{verification_url}'>here</a> to verify your email address.</p>",
        subtype="html"
    )

    # Send email
    try:
        fm = FastMail(conf)
        await fm.send_message(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email failed: {str(e)}")

    return {"message": "✅ Registered! Please verify your email."}

# Email verification
@router.get("/verify/{token}")
def verify_user(token: str):
    user = users_collection.find_one({"verificationToken": token})

    # Validate token
    if not user or user["tokenExpiry"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Token expired or invalid")

    # Mark as verified
    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"isVerified": True},
            "$unset": {"verificationToken": "", "tokenExpiry": ""}
        }
    )

    return {"message": "🎉 Email verified successfully!"}