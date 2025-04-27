# tools.py
import json
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from langchain.chat_models import ChatOpenAI
from trustcall import trustcall, TrustResult

# these globals are set by agent on startup
_llm: ChatOpenAI = None
_openapi_yaml: str = ""
_graph_states: Dict[str, Any] = {}

def set_llm_and_spec(llm: ChatOpenAI, spec: str):
    global _llm, _openapi_yaml
    _llm = llm
    _openapi_yaml = spec

def call_llm(messages) -> Any:
    """
    Helper to call the LLM with trustcall.
    Accepts a list of messages (SystemMessage/HumanMessage).
    Returns .output if success, or raises with error.
    """
    result = trustcall(_llm.invoke, messages)
    if not result.success:
        # bubble up so agent can retry or handle
        raise RuntimeError(f"LLM call failed: {result.error}")
    # result.output may be AIMessage or dict
    # unify to either .content or direct dict
    out = result.output
    if hasattr(out, "content"):
        return out.content
    if isinstance(out, dict) and "output" in out:
        return out["output"]
    return out

@tool
def general_query_fn(input: str) -> Dict[str, Any]:
    """Handle non-API questions."""
    from langchain_core.messages import SystemMessage, HumanMessage
    msgs = [
        SystemMessage(content="You are an expert API assistant."),
        HumanMessage(content=f"Answer this API query: {input}")
    ]
    content = call_llm(msgs)
    return {"intent": "general_query", "user_input": input, "response": content}

@tool
def openapi_help_fn(input: str) -> Dict[str, Any]:
    """Explain an OpenAPI endpoint given the spec."""
    from langchain_core.messages import SystemMessage, HumanMessage
    msgs = [
        SystemMessage(content="Explain the OpenAPI spec."),
        HumanMessage(content=f"Spec:\n{_openapi_yaml}\nQuestion: {input}")
    ]
    content = call_llm(msgs)
    return {"intent": "openapi_help", "user_input": input, "response": content}

@tool
def generate_payload_fn(input: str, schema: dict) -> Any:
    """Generate a realistic JSON payload matching the given schema."""
    from langchain_core.messages import SystemMessage, HumanMessage
    prompt = f"Generate JSON payload for endpoint {input} with schema {json.dumps(schema)}"
    msgs = [
        SystemMessage(content="Generate API payload."),
        HumanMessage(content=prompt)
    ]
    content = call_llm(msgs)
    try:
        return json.loads(content)
    except Exception:
        return {"error": "failed to parse payload", "raw": content}

@tool
def generate_api_execution_graph_fn() -> Dict[str, Any]:
    """Produce a dependency-aware execution graph from OpenAPI spec."""
    from langchain_core.messages import SystemMessage, HumanMessage
    prompt = f"Parse this OpenAPI spec and output JSON graph nodes & edges:\n{_openapi_yaml}"
    msgs = [
        SystemMessage(content="Create API execution graph."),
        HumanMessage(content=prompt)
    ]
    content = call_llm(msgs)
    graph = json.loads(content)
    return graph

@tool
def add_edge_fn(instruction: str) -> Dict[str, Any]:
    """Interpret user instruction to add an edge to the graph and validate."""
    from langchain_core.messages import SystemMessage, HumanMessage
    graph = _graph_states.get("default-thread", {"nodes": [], "edges": []})
    prompt = f"Graph: {json.dumps(graph)}\nInstruction: {instruction}"
    msgs = [
        SystemMessage(content="Modify the graph JSON without cycles."),
        HumanMessage(content=prompt)
    ]
    content = call_llm(msgs)
    new_graph = json.loads(content)
    _graph_states["default-thread"] = new_graph
    return new_graph

@tool
def validate_graph_fn() -> str:
    """Validate the current graph: no cycles, all nodes reachable."""
    import networkx as nx
    graph = _graph_states.get("default-thread", {})
    G = nx.DiGraph()
    for n in graph.get("nodes", []):
        G.add_node(n["operationId"])
    for e in graph.get("edges", []):
        G.add_edge(e["from_node"], e["to_node"])
    if not nx.is_directed_acyclic_graph(G):
        return "Invalid: cycles detected"
    if not nx.is_weakly_connected(G):
        return "Invalid: disconnected components"
    return "Graph valid"

@tool
def describe_execution_plan_fn() -> str:
    """Describe the execution plan from the current graph."""
    graph = _graph_states.get("default-thread", {})
    steps = [
        f"Call {e['from_node']} then {e['to_node']}"
        for e in graph.get("edges", [])
    ]
    return "Execution Plan:\n" + "\n".join(steps) if steps else "No plan defined"

@tool
def get_execution_graph_json_fn() -> Dict[str, Any]:
    """Return the current execution graph as JSON."""
    return _graph_states.get("default-thread", {"nodes": [], "edges": []})

@tool
def simulate_load_test_fn(num_users: int = 1) -> Dict[str, Any]:
    """Simulate a load test by generating concurrent execution plan for N users."""
    from langchain_core.messages import SystemMessage, HumanMessage
    graph = _graph_states.get("default-thread", {})
    prompt = f"Simulate load test on graph: {json.dumps(graph)} with {num_users} users"
    msgs = [
        SystemMessage(content="Simulate API load test."),
        HumanMessage(content=prompt)
    ]
    content = call_llm(msgs)
    try:
        return json.loads(content)
    except Exception:
        return {"error": "invalid load test output", "raw": content}
