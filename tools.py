import json
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from langchain.chat_models import ChatOpenAI

# Graph states per thread
_graph_states: Dict[str, Dict[str, Any]] = {}

_llm: ChatOpenAI

def set_llm(llm: ChatOpenAI):
    """Initialize the LLM instance for all tools."""
    global _llm
    _llm = llm

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
def generate_api_execution_graph_fn(openapi_yaml: str, input: Optional[str] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Produce a dependency-aware execution graph from OpenAPI spec."""
    prompt = f"Parse this OpenAPI spec and output JSON graph nodes & edges:\n{openapi_yaml}"
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    graph = json.loads(content)
    # Persist per-thread graph state
    if thread_id:
        _graph_states[thread_id] = graph
    return graph

@tool
def add_edge_fn(input: str, openapi_yaml: str, thread_id: str) -> Dict[str, Any]:
    """Interpret user instruction to add an edge, validate with LLM, update graph."""
    graph = _graph_states.get(thread_id, {"nodes": [], "edges": []})
    prompt = (
        f"Graph: {json.dumps(graph)}\n"
        f"Spec: {openapi_yaml}\nInstruction: {input}\n"
        "Modify the graph JSON accordingly without cycles. Return only JSON."
    )
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    graph = json.loads(content)
    _graph_states[thread_id] = graph
    return graph

@tool
def validate_graph_fn(thread_id: str) -> str:
    """Validate current graph: no cycles, all nodes reachable."""
    import networkx as nx
    graph = _graph_states.get(thread_id, {})
    G = nx.DiGraph()
    for n in graph.get("nodes", []):
        G.add_node(n["operationId"])
    for e in graph.get("edges", []):
        G.add_edge(e["from_node"], e["to_node"])
    issues = []
    if not nx.is_directed_acyclic_graph(G):
        issues.append("cycles detected")
    if not nx.is_weakly_connected(G):
        issues.append("disconnected components")
    return "Invalid: " + ", ".join(issues) if issues else "Graph valid"

@tool
def describe_execution_plan_fn(thread_id: str) -> str:
    """Describe the execution plan from the current graph."""
    graph = _graph_states.get(thread_id, {})
    steps = [
        f"Call {e['from_node']} then {e['to_node']}" 
        for e in graph.get("edges", [])
    ]
    return "Execution Plan:\n" + "\n".join(steps) if steps else "No plan defined"

@tool
def get_execution_graph_json_fn(thread_id: str) -> Dict[str, Any]:
    """Return the current execution graph as JSON."""
    return _graph_states.get(thread_id, {"nodes": [], "edges": []})

@tool
def simulate_load_test_fn(thread_id: str, num_users: int = 1) -> Dict[str, Any]:
    """Simulate a load test by generating concurrent execution plan for N users."""
    graph = _graph_states.get(thread_id, {})
    # Use LLM to generate load test plan
    prompt = (
        f"Graph for load test: {json.dumps(graph)}\n"
        f"Simulate load with {num_users} users and output stats JSON."
    )
    resp = _llm.invoke({"input": prompt})
    content = resp.get("output") or (resp.get("messages") or [])[-1].content
    try:
        return json.loads(content)
    except Exception:
        return {"error": "invalid load test output", "raw": content}
