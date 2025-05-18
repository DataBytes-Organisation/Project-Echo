from fastapi import APIRouter, HTTPException, status
from typing import List
from app.database import Nodes
from datetime import datetime

router = APIRouter()

@router.get("/nodes", response_description="List all nodes")
def get_nodes():
    try:
        nodes = list(Nodes.find({}))
        return nodes
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving nodes: {str(error)}")

@router.get("/nodes/{node_id}", response_description="Get a single node by ID")
def get_node(node_id: str):
    try:
        node = Nodes.find_one({"_id": node_id})
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
            
        # If node has connected nodes, fetch their basic info
        if node.get('connectedNodes') and len(node['connectedNodes']) > 0:
            connected_nodes = list(Nodes.find(
                {"_id": {"$in": node['connectedNodes']}},
                {"name": 1, "type": 1, "model": 1, "location": 1}
            ))
            node['connectedNodesData'] = connected_nodes
            
        return node
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving node: {str(error)}")

@router.get("/nodes/{node_id}/components", response_description="Get components for a node")
def get_node_components(node_id: str):
    try:
        node = Nodes.find_one({"_id": node_id})
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
            
        return node.get('components', [])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving node components: {str(error)}")

@router.get("/nodes/{node_id}/connections", response_description="Get connections for a node")
def get_node_connections(node_id: str):
    try:
        node = Nodes.find_one({"_id": node_id})
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
            
        # Get connected nodes with basic info
        connected_nodes = list(Nodes.find(
            {"_id": {"$in": node.get('connectedNodes', [])}},
            {"name": 1, "type": 1, "model": 1, "location": 1}
        ))
        
        return {"connectedNodes": connected_nodes}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving node connections: {str(error)}")

@router.put("/nodes/{node_id}/register", response_description="Register a node")
def register_node(node_id: str):
    try:
        result = Nodes.update_one(
            {"_id": node_id},
            {"$set": {
                "registered": True,
                "registeredTime": datetime.utcnow().isoformat() + "+00:00"
            }}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Node not found")
            
        if result.modified_count == 1:
            return {"message": "Node registered successfully"}
        else:
            return {"message": "Node was already registered"}
            
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error registering node: {str(error)}")
