## app.middleware.auth.py
import jwt
# from typing import Dict, List
import datetime
from bson.objectid import ObjectId

import logging

from app.config import settings

logger = logging.getLogger(__name__)

JWT_SECRET = settings.jwt_secret
JWT_ALGORITHM = settings.jwt_algorithm


def token_response(token: str):
    return {
        "access_token": token
    }

def signJWT(user: dict, authorities: list[str]) -> str:
    payload = {
        "id": str(user["_id"]),
        "roles": authorities,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=settings.jwt_expiry_seconds)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token

# Handle JWT token received from HMI
def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        logger.debug("JWT decoded successfully")
        token_is_valid = (
            datetime.datetime.utcfromtimestamp(decoded_token["exp"])
            >= datetime.datetime.utcnow()
        )

        return decoded_token if token_is_valid else None

    except Exception:
        logger.warning("JWT decoding failed", exc_info=True)
        return None