import jwt
from decouple import config
from typing import Dict, Str, List
import datetime


JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")


def token_response(token: str):
    return {
        "access_token": token
    }

def signJWT(user: dict, authorities: List[str]) -> str:
    payload = {
        "id": user["userId"],
        "roles": authorities,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=86400)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token

# Handle JWT token received from HMI
def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, [JWT_SECRET], algorithms = [JWT_ALGORITHM])
        return decoded_token if decoded_token["expires"] >= datetime.datetime.utcnow() else None
    except:
        return {}