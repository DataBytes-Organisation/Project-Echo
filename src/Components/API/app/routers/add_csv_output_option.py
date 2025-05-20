from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from datetime import datetime
import io
import csv

from .. import database  # Imports MongoDB collections from database.py

router = APIRouter()

@router.get("/events/csv")
async def export_events_csv(
    species: str = None,
    mic_id: str = None,
    start_time: str = None,
    end_time: str = None
):
    # Build the MongoDB query
    query = {}
    if species:
        query["species"] = species
    if mic_id:
        query["mic_id"] = mic_id
    if start_time and end_time:
        query["timestamp"] = {
            "$gte": datetime.fromisoformat(start_time),
            "$lte": datetime.fromisoformat(end_time)
        }

    # Query the database
    results = database.Events.find(query)

    # Build the CSV output
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "species", "mic_id"])  # Change if needed

    for doc in results:
        writer.writerow([
            doc.get("timestamp", ""),
            doc.get("species", ""),
            doc.get("mic_id", "")
        ])

    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=events_data.csv"
    })
