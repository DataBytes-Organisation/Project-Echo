from typing import Optional
from fastapi import APIRouter, Body, Path, Query

from app.schemas import (
    Project,
    ProjectCreate,
    ProjectUpdate,
    ProjectListResponse,
)
from app.services import projects as projects_service

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=ProjectListResponse, summary="List projects")
def get_projects(
    status: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    ecologist: Optional[str] = Query(None),
):
    return projects_service.list_projects(status=status, location=location, ecologist=ecologist)


@router.post("", response_model=Project, summary="Create a project")
def create_project(payload: ProjectCreate = Body(...)):
    return projects_service.create_project(payload)


@router.put("/{project_id}", response_model=Project, summary="Update a project (replace)")
def update_project(
    project_id: str = Path(..., description="Mongo ObjectId"),
    payload: ProjectUpdate = Body(...),
):
    return projects_service.update_project(project_id, payload)


@router.delete("/{project_id}", summary="Delete a project")
def delete_project(project_id: str = Path(..., description="Mongo ObjectId")):
    return projects_service.delete_project(project_id)


# Optional endpoint
@router.get("/_meta/ecologists", summary="List ecologists")
def get_ecologists():
    return {"ecologists": projects_service.list_ecologists()}
