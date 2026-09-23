from app.database import Events
from app.schemas import EventSchema


def persist_event(event: EventSchema):
    """
    Persist a validated Engine/Backend event in the shared Events collection.

    HTTP ingestion and MQTT ingestion both use this function so event
    persistence follows one Backend path.
    """
    result = Events.insert_one(event.dict())
    return result.inserted_id