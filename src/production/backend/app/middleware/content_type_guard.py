from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class ContentTypeGuardMiddleware(BaseHTTPMiddleware):
    """
    Middleware that intercepts POST and PUT requests.
    If the request has a body (indicated by content-length or transfer-encoding),
    it enforces that the Content-Type must be either application/json or multipart/form-data.
    Otherwise, it rejects the request with a 415 Unsupported Media Type.
    """
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT"]:
            content_length = request.headers.get("content-length")
            transfer_encoding = request.headers.get("transfer-encoding")
            
            # Check if there is a body
            if (content_length and int(content_length) > 0) or transfer_encoding == "chunked":
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith("application/json") and not content_type.startswith("multipart/form-data") and not content_type.startswith("application/x-www-form-urlencoded"):
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={"detail": f"Unsupported media type: {content_type}. Must be application/json, multipart/form-data, or application/x-www-form-urlencoded."}
                    )
        
        response = await call_next(request)
        return response
