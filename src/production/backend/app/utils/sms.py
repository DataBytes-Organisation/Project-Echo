import logging

from twilio.rest import Client

from app.config import settings

logger = logging.getLogger(__name__)


def send_sms(to_number: str, message: str) -> bool:
    """
    Send SMS using Twilio

    Args:
        to_number (str): Recipient's phone number in E.164 format (e.g., +61412345678)
        message (str): Message content

    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    if not (settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_phone_number):
        logger.warning("Twilio settings are not configured; skipping SMS send")
        return False

    try:
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        client.messages.create(
            body=message,
            from_=settings.twilio_phone_number,
            to=to_number
        )
        return True
    except Exception:
        logger.warning("Error sending SMS", exc_info=True)
        return False
