"""Similar-detection retrieval endpoint.

Given a detection that already has a stored model embedding (see
app/schemas.py's DetectionCreate.embedding), finds the most similar past
detections by cosine similarity, so a reviewer looking at an uncertain
detection has precedent to compare against instead of judging it from
confidence alone. Also surfaces two signals cheaply computed from the same
lookup: `ambiguous` (the nearest neighbours don't agree on a species) and
`novel` (nothing in the database looks like this at all).
"""

from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query

from app.database import Detections
from app.services.similarity import top_k_similar, flag_result

router = APIRouter(
    prefix="/detections",
    tags=["detections"],
)


def _load_candidates(exclude_id: str) -> list:
    candidates = []
    cursor = Detections.find(
        {"embedding": {"$exists": True, "$ne": None}},
        {"species": 1, "embedding": 1},
    )
    for doc in cursor:
        doc_id = str(doc["_id"])
        if doc_id == exclude_id:
            continue
        candidates.append({"id": doc_id, "species": doc.get("species"), "embedding": doc.get("embedding")})
    return candidates


@router.get("/{detection_id}/similar", summary="Find past detections with the closest model embedding")
def similar_detections_endpoint(detection_id: str, k: int = Query(5, ge=1, le=20)) -> Dict[str, Any]:
    try:
        oid = ObjectId(detection_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid detection id")

    query_doc = Detections.find_one({"_id": oid})
    if not query_doc:
        raise HTTPException(status_code=404, detail="Detection not found")

    query_embedding = query_doc.get("embedding")
    if not query_embedding:
        raise HTTPException(status_code=422, detail="This detection has no stored embedding to compare with")

    candidates = _load_candidates(exclude_id=detection_id)
    top_k = top_k_similar(query_embedding, candidates, k=k)
    flags = flag_result(top_k)

    return {
        "detection_id": detection_id,
        "results": [
            {"detection_id": candidate["id"], "species": candidate["species"], "similarity": round(score, 4)}
            for score, candidate in top_k
        ],
        **flags,
    }
