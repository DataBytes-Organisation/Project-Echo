from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import decodeJWT
from bson import ObjectId

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
    def verify_role(self, role: str) -> (bool, str):
        res = None
        try:
            print("JWT Verification: {}".format(self.isVerified))
            print("UserInfo: {}".format(self.decodedUser))
            if self.isVerified:
                print("decoded userInfo: {}".format(self.decodedUser))
                res = [i for i in self.decodedUser["roles"] if role.upper() in i.upper()]
                if res == None:
                    return (False, "User does not have the role {}".format(role))
                else:
                    return (True, "User role is validated")

        except:
            return (False, "An Error occured when validate role")

   