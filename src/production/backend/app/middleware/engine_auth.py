## app.middleware.engine_auth.py
"""Service-to-service authentication for Engine -> Backend calls.

Separate from JWTBearer (app.middleware.auth_bearer), which authenticates a
logged-in HMI user against a role. The Engine is a machine client with no
user account, so it authenticates with a static API key instead.
"""

import os

from fastapi import Header, HTTPException

ENGINE_API_KEY_ENV = "ENGINE_API_KEY"
ENGINE_API_KEY_HEADER = "x-engine-api-key"


def verify_engine_api_key(x_engine_api_key: str = Header(default=None)):
    expected = os.getenv(ENGINE_API_KEY_ENV)
    if not expected:
        # No key provisioned for this environment (e.g. local/dev) - leave the
        # endpoint open rather than locking out everyone who hasn't set one.
        return
    if x_engine_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing Engine API key.")
