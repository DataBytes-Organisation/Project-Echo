from fastapi import APIRouter, HTTPException
import json
import os
from app.serializers import nodeListEntity

router = APIRouter()

@router.get("/", summary="Get All Nodes")
def get_all_nodes():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "nodes.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        return nodeListEntity(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
