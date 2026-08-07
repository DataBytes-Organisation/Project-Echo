"""Test configuration for the backend unit tests.

The backend application lives at src/production/backend, which is not on
Python's import path when pytest is run from the repository root. This file
adds it, so the tests can do ``from app import config``.

pytest imports conftest.py automatically before collecting tests in this
directory, so nothing needs to import this module explicitly.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[3] / "production" / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
