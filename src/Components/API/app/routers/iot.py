from fastapi import status, APIRouter
from app.database import Nodes, Components, Commands

from bson.objectid import ObjectId
import datetime

router = APIRouter()

def get_node(node_id):
    """
    Retrieve a node from MongoDB by its id.
    """
    try:
        node = Nodes.find_one({"_id": ObjectId(node_id)})
    except Exception:
        return None
    return node

def get_component(component_id):
    """
    Retrieve a component from MongoDB by its id.
    """
    try:
        component = Components.find_one({"_id": ObjectId(component_id)})
    except Exception:
        return None
    return component

def create_command(target_type, target_id, command_type, parameters):
    """
    Log a command in the commands collection.
    The target_type can be "node" or "component".
    """
    command = {
        "target_type": target_type,
        "target_id": ObjectId(target_id),
        "command_type": command_type,
        "parameters": parameters,
        "status": "pending",  # Updated later once command is dispatched/received
        "issued_at": datetime.datetime.utcnow()
    }
    command_id = Commands.insert_one(command).inserted_id
    return str(command_id)

def dispatch_command(target_id, command_type, parameters, target_type="node"):
    """
    Placeholder function to send a command to a node or component.
    In a real system, this could interface with an MQTT broker, WebSocket, etc.
    """
    print(f"Dispatching command '{command_type}' with parameters {parameters} to {target_type} {target_id}")
    
    return True

