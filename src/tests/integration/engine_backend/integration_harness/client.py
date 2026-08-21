"""Send a Project Echo detection event to the Backend API."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request


DEFAULT_BACKEND_URL = "http://localhost:9000/engine/event"
DEFAULT_TIMEOUT_SECONDS = 5.0


class HarnessError(RuntimeError):
    """Base error raised by the Engine-Backend integration client."""


class PayloadError(HarnessError):
    """Raised when a payload file cannot be used."""


class BackendRequestError(HarnessError):
    """Raised when the Backend rejects a request or cannot be reached."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class BackendResponse:
    """Successful response returned by the Backend."""

    status_code: int
    body: Any


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load one JSON object from a fixture file."""

    payload_path = Path(path)
    try:
        with payload_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError as exc:
        raise PayloadError(f"Payload file not found: {payload_path}") from exc
    except json.JSONDecodeError as exc:
        raise PayloadError(f"Payload file is not valid JSON: {payload_path}") from exc

    if not isinstance(payload, dict):
        raise PayloadError("The payload must be a JSON object.")
    return payload


def send_detection(
    backend_url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> BackendResponse:
    """POST a detection and accept only the current Backend success code, 201."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    encoded_payload = json.dumps(dict(payload)).encode("utf-8")
    http_request = request.Request(
        backend_url,
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
            try:
                body = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                body = raw_body

            if response.status != 201:
                raise BackendRequestError(
                    f"Unexpected Backend status {response.status}",
                    status_code=response.status,
                )
            return BackendResponse(status_code=response.status, body=body)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise BackendRequestError(
            f"Backend returned HTTP {exc.code}: {details}",
            status_code=exc.code,
        ) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise BackendRequestError(
            f"Backend request timed out after {timeout_seconds} seconds"
        ) from exc
    except error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise BackendRequestError(
                f"Backend request timed out after {timeout_seconds} seconds"
            ) from exc
        raise BackendRequestError(f"Could not connect to Backend: {exc.reason}") from exc


def main() -> int:
    """Run one manual Engine-to-Backend integration request."""

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "valid_payloads"
        / "valid_detection_01.json"
    )
    parser = argparse.ArgumentParser(
        description="Send a Project Echo detection fixture to the Backend."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("ECHO_BACKEND_URL", DEFAULT_BACKEND_URL),
        help="Backend event endpoint",
    )
    parser.add_argument("--payload", default=str(fixture), help="JSON payload file")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    try:
        payload = load_payload(args.payload)
        response = send_detection(args.url, payload, args.timeout)
    except (HarnessError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS: Backend returned HTTP {response.status_code}")
    print(json.dumps(response.body, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
