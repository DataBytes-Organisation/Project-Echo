## app.middleware.auth_bearer.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import decodeJWT
from bson import ObjectId

import logging

logger = logging.getLogger(__name__)

class JWTBearer(HTTPBearer):
    isVerified = False
    decodedUser = None

    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            
            #VerifyJWTToken return bool and payload
            #Only need bool value
            (self.isVerified, self.decodedUser) = self.verify_jwt(credentials.credentials)
            if not self.isVerified:
                raise HTTPException(status_code=403, detail="Invalid token or expired token.")
            #For now, return credentials when pass bearer
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    #Verify the JWT token, return both result and decoded payload
    def verify_jwt(self, JWTToken: str) -> (bool, dict):
        isTokenValid: bool = False

        try:
            payload = decodeJWT(JWTToken)
        except:
            payload = None
        if payload != None:
            isTokenValid = True
        return (isTokenValid, payload)
    
    #Verify user role using JWT token
    def verify_role(self, role: str) -> tuple[bool, str]:
        try:
            logger.debug("Starting JWT role verification")

            if not self.isVerified:
                logger.warning("Role verification attempted with an unverified token")
                return False, "Token is not verified"

            matching_roles = [
                user_role
                for user_role in self.decodedUser.get("roles", [])
                if role.upper() in user_role.upper()
            ]

            if not matching_roles:
                logger.warning("User role validation failed")
                return False, f"User does not have the role {role}"

            logger.info("User role validated successfully")
            return True, "User role is validated"

        except Exception:
            logger.exception("An error occurred while validating the user role")
            return False, "An error occurred when validating role"