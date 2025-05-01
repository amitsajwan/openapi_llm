import os
import json
from typing import Dict, Any

def load_state(graph_name: str, session_id: str) -> Dict[str, Any]:
    state_file = f"{graph_name}_{session_id}.json"
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}

def save_state(state: Dict[str, Any], graph_name: str, session_id: str):
    state_file = f"{graph_name}_{session_id}.json"
    with open(state_file, 'w') as f:
        json.dump(state, f)
