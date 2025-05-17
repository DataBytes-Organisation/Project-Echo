from twilio.rest import Client
from decouple import config

# Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = "AC8f3e6e91bd33bdda1dd2e453327e6d3a"
TWILIO_AUTH_TOKEN = "ecb3f62b59de94d7158150b961e73ec9"
TWILIO_PHONE_NUMBER = "+19035013811"

def send_sms(to_number: str, message: str) -> bool:
    """
    Send SMS using Twilio
    
    Args:
        to_number (str): Recipient's phone number in E.164 format (e.g., +61412345678)
        message (str): Message content
        
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print(to_number)
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        return True
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")
        return False
