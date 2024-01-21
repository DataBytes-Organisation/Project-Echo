import jwt
from decouple import config
# from typing import Dict, List
import datetime
from bson.objectid import ObjectId

JWT_SECRET = "deff1952d59f883ece260e8683fed21ab0ad9a53323eca4f"
JWT_ALGORITHM = "HS256"


def token_response(token: str):
    return {
        "access_token": token
    }

def signJWT(user: dict, authorities: list[str]) -> str:
    payload = {
        "id": str(user["_id"]),
        "roles": authorities,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=86400)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token

# Handle JWT token received from HMI
def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms = [JWT_ALGORITHM])
        print("toggled decode function! The result is: {}".format(decoded_token))
        return decoded_token if datetime.datetime.utcfromtimestamp(decoded_token["exp"]) >= datetime.datetime.utcnow() else None
    except Exception as e:
        print("Decode failed! Need to check on this step: {}".format(e))
        return None