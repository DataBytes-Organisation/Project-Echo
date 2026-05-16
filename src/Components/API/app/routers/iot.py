from fastapi import APIRouter, HTTPException, status, Request, Body
from typing import List, Optional
from app.database import Nodes
from datetime import datetime
from pymongo import UpdateOne

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

@router.put("/nodes/{node_id}/heartbeat", response_description="Update node's last message and connection status")
async def update_node_connection(node_id: str, request: Request):
    try:
        # Store time with timezone offset
        current_time = datetime.now().astimezone().isoformat()
        update_fields = {"lastSeen": current_time}
        
        # Try to get message from request body if it exists
        try:
            body = await request.json()
            if "message" in body:
                update_fields["lastMessage"] = body["message"]
        except:
            # No body or invalid JSON - just update lastSeen
            pass
            
        result = Nodes.update_one(
            {"_id": node_id},
            {"$set": update_fields}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Node not found")
            
        if result.modified_count == 1:
            return {"message": "Node connection status updated successfully"}
        else:
            return {"message": "Node connection status unchanged"}
            
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error updating node connection: {str(error)}")



@router.post("/nodes/{node_id}/updates", response_description="Update node's components data")
async def update_node_data(node_id: str, request: Request):
    try:
        # Get update data from request body
        try:
            body = await request.json()
            if not body or not isinstance(body, list):
                raise HTTPException(status_code=400, detail="Request body must be an array of component updates")
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
            
        # Validate each component update
        for update in body:
            if not isinstance(update, dict) or 'component_id' not in update or 'data' not in update:
                raise HTTPException(status_code=400, detail="Each update must have component_id and data fields")
            if not isinstance(update['data'], dict):
                raise HTTPException(status_code=400, detail="Data field must be an object")
        
        # Prepare bulk write operations
        operations = []
        print(body)
        for update in body:
            component_id = update['component_id']
            sensor_data = update['data']
            
            # Create an update operation for each component
            operations.append(
                UpdateOne(
                    {
                        "_id": node_id,
                        "components.id": component_id
                    },
                    {
                        "$set": {
                            "components.$.sensorData": sensor_data
                        }
                    }
                )
            )
        
        # Execute all updates in one bulk operation
        result = Nodes.bulk_write(operations)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Node or components not found")
            
        return {
            "message": f"Updated {result.modified_count} components",
            "modified_count": result.modified_count,
            "matched_count": result.matched_count
        }
            
    except HTTPException as http_error:
        raise http_error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error updating component data: {str(error)}")
