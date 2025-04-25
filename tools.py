# File: tools.py
import json
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from langchain.chat_models import ChatOpenAI

# Shared LLM instance inside tools
_llm = ChatOpenAI(model="gpt-4", temperature=0)

# In-memory graph state
_graph_state: Dict[str, Any] = {"nodes": [], "edges": []}

@tool
def general_query_fn(input: str, openapi_yaml: Optional[str] = None) -> Dict[str, Any]:
    """Answer any general API-related question using LLM reasoning."""
    resp = _llm.invoke({"input": f"Answer this API query: {input}"})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    return {"intent": "general_query", "user_input": input, "response": content}

@tool
def openapi_help_fn(input: str, openapi_yaml: str) -> Dict[str, Any]:
    """Explain what an OpenAPI endpoint does given the spec."""
    prompt = f"Explain this OpenAPI spec:\n{openapi_yaml}\nQuestion: {input}" 
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    return {"intent": "openapi_help", "user_input": input, "response": content}

@tool
def generate_payload_fn(input: str, schema: dict, openapi_yaml: Optional[str] = None) -> Any:
    """Generate a realistic JSON payload matching the given schema."""
    prompt = f"Generate JSON payload for endpoint {input} with schema {json.dumps(schema)}"
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    try:
        return json.loads(content)
    except Exception:
        return {"error": "failed to parse payload", "raw": content}

@tool
def generate_api_execution_graph_fn(openapi_yaml: str, input: Optional[str] = None) -> Dict[str, Any]:
    """Produce a dependency-aware execution graph from OpenAPI spec."""
    prompt = f"Parse this OpenAPI spec and output JSON graph nodes & edges:\n{openapi_yaml}"  
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    graph = json.loads(content)
    _graph_state.clear()
    _graph_state.update(graph)
    return graph

@tool
def add_edge_fn(input: str, openapi_yaml: str) -> Dict[str, Any]:
    """Interpret user instruction to add an edge to the graph and validate with LLM."""
    graph_str = json.dumps(_graph_state)
    prompt = f"Graph: {graph_str}\nSpec: {openapi_yaml}\nInstruction: {input}\nModify the graph JSON accordingly without cycles. Return only JSON." 
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    graph = json.loads(content)
    _graph_state.clear()
    _graph_state.update(graph)
    return graph

@tool
def validate_graph_fn(openapi_yaml: Optional[str] = None, input: Optional[str] = None) -> str:
    """Validate current graph: no cycles, all nodes reachable."""
    import networkx as nx
    G = nx.DiGraph()
    for n in _graph_state.get("nodes", []): G.add_node(n.get("operationId"))
    for e in _graph_state.get("edges", []): G.add_edge(e.get("from"), e.get("to"))
    if not nx.is_directed_acyclic_graph(G): return "Invalid: cycles detected"
    if not nx.is_weakly_connected(G): return "Invalid: disconnected components"
    return "Graph valid"

@tool
def describe_execution_plan_fn(openapi_yaml: Optional[str] = None, input: Optional[str] = None) -> str:
    """Describe the execution plan from the current graph."""
    steps = [f"Call {e['from']} → then {e['to']}" for e in _graph_state.get("edges", [])]
    return "Execution Plan:\n" + "\n".join(steps) if steps else "No plan defined"

@tool
def get_execution_graph_json_fn(openapi_yaml: Optional[str] = None, input: Optional[str] = None) -> Dict[str, Any]:
    """Return the current execution graph as JSON."""
    return _graph_state
