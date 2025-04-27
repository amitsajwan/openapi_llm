# tools.py
import json
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from trustcall import create_extractor
from langchain.chat_models import ChatOpenAI

# --- Globals to be set by agent startup ---
_llm: ChatOpenAI = None
_openapi_yaml: str = ""
_graph_state: Dict[str, Any] = {"nodes": [], "edges": []}

# --- Helper to bind LLM+tools into trustcall extractor ---
extractor = None
def bind_trustcall(llm, tools: List[Any]):
    global extractor
    extractor = create_extractor(
        llm,
        tools=tools,
        tool_choice="any",          # let TrustCall pick the right tool
    )

# --- Your tools unchanged, except they assume global spec & state ---
@tool
def get_api_list() -> List[str]:
    """List all endpoints in the OpenAPI spec."""
    return list(_openapi_yaml.get("paths", {}).keys())

@tool
def determine_api_sequence(endpoints: List[str]) -> List[str]:
    """Sort or logically order endpoints for execution."""
    return sorted(endpoints)

@tool
def generate_graph(ordered_endpoints: List[str]) -> Dict[str, Any]:
    """Generate nodes & edges from an ordered list of endpoints."""
    nodes = [{"id": e} for e in ordered_endpoints]
    edges = [{"from": ordered_endpoints[i], "to": ordered_endpoints[i+1]}
             for i in range(len(ordered_endpoints)-1)]
    return {"nodes": nodes, "edges": edges}

# ... include your other tools (add_edge_fn, validate_graph_fn, etc.) similarly ...
