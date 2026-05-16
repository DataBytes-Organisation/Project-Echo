## app.routers.sim.py
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List
from app import serializers
from app import schemas
from app.database import Movements, Microphones, User
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import asyncio
from datetime import datetime, timedelta

router = APIRouter()
# Email configuration for sending notifications
conf = ConnectionConfig(
    MAIL_USERNAME="projectechodeakintest@gmail.com",
    MAIL_PASSWORD="oocr srvw ndoj bwte",  #App password
    MAIL_FROM="projectechodeakintest@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    # MAIL_STARTTLS=True,
    # MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

# Function to send email notifications
async def send_email_notification(user, species):
    print("USER IS ", user)
    now = datetime.now()
    user_email = user['email']

    # Check if the user has received an email for this species
    email_notifications = user.get('emailNotifications', [])

    for notification in email_notifications:
        last_sent = notification['last_sent']

        # Convert last_sent to datetime object if it's a string
        if isinstance(last_sent, str):
            last_sent = datetime.fromisoformat(last_sent)

        if notification['species'] == species:
            # Check if the last email was sent more than an hour ago
            if now - last_sent < timedelta(hours=1):
                print(f"Email already sent to {user_email} for {species} within the last hour.")
                return

    # Create email message with some content
    message = MessageSchema(
        subject="Animal Detected",
        recipients=[user_email],
        body=f"An animal of species {species} has been detected near your monitoring location.",
        subtype="plain"
    )

    # Attempt to send the email
    try:
        print(f"Sending email to {user_email} for species {species}")
        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"Email sent to {user_email} for animal {species} detected.")
    except Exception as e:
        print(f"Failed to send email to {user_email}. Error: {e}")
        return

    # Update the user's emailNotifications field
    for notification in email_notifications:
        if notification['species'] == species:
            notification['last_sent'] = now.isoformat()
            break
    else:
        # If the species is not in the list, add it
        email_notifications.append({'species': species, 'last_sent': now.isoformat()})

    # Update the user document in the database
    try:
        User.update_one(
            {"_id": user['_id']},
            {"$set": {"emailNotifications": email_notifications}}
        )
        print(f"Updated emailNotifications for user {user['_id']}")
    except Exception as e:
        print(f"Failed to update user document: {e}")


# Function to check user subscriptions and notify
def check_and_notify_users(species_detected):
    # Query users who want to be notified about this species
    users_to_notify = User.find({
        "notificationAnimals": {
            "$elemMatch": {"species": species_detected}
        }
    })


    # Notify each user
    for user in users_to_notify:
        print("Users to notify", user)
        user_email = user['email']
        # Send email asynchronously
        asyncio.run(send_email_notification(user, species_detected))


# Simulated function to cr

@router.post("/movement", status_code=status.HTTP_201_CREATED)
def create_movement(movement: schemas.MovementSchema):

    result = Movements.insert_one(movement.dict())
    pipeline = [
            {'$match': {'_id': result.inserted_id}},
        ]
    new_post = serializers.movementListEntity(Movements.aggregate(pipeline))[0]
    print("new post",new_post)
    check_and_notify_users(new_post['species'])

    return new_post

@router.post("/microphones", status_code=status.HTTP_201_CREATED)
def create_microphones(microphones: List[schemas.MicrophoneSchema]):
    Microphones.drop()
    microphone_list = []
    for microphone in microphones:
        print(microphone)

        Microphones.insert_one(microphone.dict())
        microphone_list.append(microphone.dict())
        
    return JSONResponse(content = microphone_list)