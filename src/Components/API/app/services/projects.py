from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import HTTPException
from bson import ObjectId
from pymongo import ReturnDocument

from app.database import Projects
from app.schemas import ProjectCreate, ProjectUpdate, Project, ProjectListResponse


def _to_project(doc: Dict[str, Any]) -> Project:
    return Project(
        id=str(doc["_id"]),
        name=doc.get("name", ""),
        description=doc.get("description"),
        location=doc.get("location"),
        status=doc.get("status", ""),
        sensorIds=doc.get("sensorIds", []),
        ecologists=doc.get("ecologists", []),
    )


def list_projects(
    status: Optional[str] = None,
    location: Optional[str] = None,
    ecologist: Optional[str] = None,
) -> ProjectListResponse:
    q: Dict[str, Any] = {}
    if status:
        q["status"] = status
    if location:
        q["location"] = location
    if ecologist:
        q["ecologists"] = ecologist

    total = Projects.count_documents(q)
    items = [_to_project(d) for d in Projects.find(q).sort("name", 1)]
    return ProjectListResponse(items=items, total=total)


def create_project(payload: ProjectCreate) -> Project:
    doc = payload.model_dump()
    now = datetime.now(timezone.utc)
    doc["created_at"] = now
    doc["updated_at"] = now

    res = Projects.insert_one(doc)
    created = Projects.find_one({"_id": res.inserted_id})
    return _to_project(created)


def update_project(project_id: str, payload: ProjectUpdate) -> Project:
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project id")

    now = datetime.now(timezone.utc)
    new_doc = payload.model_dump()
    new_doc["updated_at"] = now

    updated = Projects.find_one_and_update(
        {"_id": ObjectId(project_id)},
        {"$set": new_doc},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")

    return _to_project(updated)


def delete_project(project_id: str) -> Dict[str, Any]:
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project id")

    res = Projects.delete_one({"_id": ObjectId(project_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"deleted": True}


def list_ecologists() -> List[str]:
    pipeline = [
        {"$unwind": "$ecologists"},
        {"$group": {"_id": "$ecologists"}},
        {"$sort": {"_id": 1}},
    ]
    rows = list(Projects.aggregate(pipeline))
    return [r["_id"] for r in rows if r.get("_id")]
