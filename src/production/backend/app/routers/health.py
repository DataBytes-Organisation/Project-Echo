from fastapi import APIRouter
from app.database import db

router = APIRouter()

@router.get("")
def health_check():
    """
    Returns the health status of the API and the MongoDB connection.
    """
    db_connected = False
    try:
        # A simple ping command to check if the database is responsive
        db.command("ping")
        db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "ok",
        "db_connected": db_connected
    }
