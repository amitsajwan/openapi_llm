"""
state_manager.py

Provides functions to load and save the LangGraph execution state using MemorySaver or file-based persistence.

Functions:
  - load_state(graph_name: str, session_id: Optional[str] = None) -> Dict[str, Any]
  - save_state(state: Dict[str, Any], graph_name: str, session_id: Optional[str] = None) -> None

Integrates with langgraph.checkpoint.memory.MemorySaver for in-graph checkpointing.
"""
from typing import Any, Dict, Optional
import os
import json
from langgraph.checkpoint.memory import MemorySaver

# default folder for file-based backups
BACKUP_DIR = os.path.expanduser("~/.openapi_state_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

# instantiate a single MemorySaver for all graphs
_memory_saver = MemorySaver()


def load_state(graph_name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the saved state for a given graph and session.
    First tries MemorySaver, then falls back to file-based JSON.

    Args:
      graph_name: name of the StateGraph
      session_id: optional session identifier
    Returns:
      state dict (may be empty if no prior state)
    """
    # attempt in-memory load
    saved = _memory_saver.load(graph_name=graph_name, session_id=session_id)
    if saved is not None:
        return saved

    # fallback to file
    fname = f"{graph_name}.{session_id or 'default'}.json"
    path = os.path.join(BACKUP_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(state: Dict[str, Any], graph_name: str, session_id: Optional[str] = None) -> None:
    """
    Save the state dict for a given graph and session.
    Writes to MemorySaver and also to a file backup.

    Args:
      state: dict of state values
      graph_name: name of the StateGraph
      session_id: optional session identifier
    """
    # save in-memory for fast retrieval
    _memory_saver.save(state, graph_name=graph_name, session_id=session_id)

    # also write a file backup
    fname = f"{graph_name}.{session_id or 'default'}.json"
    path = os.path.join(BACKUP_DIR, fname)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
