from starlette.datastructures import Headers
from fastapi.responses import JSONResponse


class RequestTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(
        self,
        app,
        json_limit: int = 1 * 1024 * 1024,
        upload_limit: int = 32 * 1024 * 1024,
    ):
        self.app = app
        self.json_limit = json_limit
        self.upload_limit = upload_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")

        # Only methods normally containing request bodies
        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_type = headers.get("content-type", "")

        if content_type.startswith("multipart/form-data"):
            max_size = self.upload_limit
        else:
            max_size = self.json_limit

        # Reject immediately if Content-Length already proves it is too large
        content_length = headers.get("content-length")

        if content_length:
            try:
                if int(content_length) > max_size:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload Too Large",
                            "message": "Request body exceeds the allowed size."
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        received_size = 0

        async def limited_receive():
            nonlocal received_size

            message = await receive()

            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))

                if received_size > max_size:
                    raise RequestTooLarge()

            return message

        try:
            await self.app(scope, limited_receive, send)

        except RequestTooLarge:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": "Payload Too Large",
                    "message": "Request body exceeds the allowed size."
                },
            )
            await response(scope, receive, send)