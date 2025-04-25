from typing import Any, List
from langchain_core.tools import tool

# In-memory graph state
_graph_state = {"nodes": [], "edges": []}

@tool
def general_query_fn(query: str) -> str:
    """Answer any general API-related question using LLM reasoning."""
    return f"General answer for '{query}'"

@tool
def openapi_help_fn(question: str, openapi_yaml: str) -> str:
    """Explain what an OpenAPI endpoint does given the spec."""
    return f"Explanation for '{question}' from the OpenAPI spec."

@tool
def generate_payload_fn(endpoint: str, schema: dict) -> dict:
    """Generate a JSON payload matching the given schema."""
    # Example implementation for payload generation
    return {key: f"example_{key}" for key in schema.keys()}

@tool
def generate_api_execution_graph_fn(openapi_yaml: str) -> dict:
    """Produce a dependency-aware execution graph (nodes & edges) from OpenAPI spec."""
    # Example graph generation
    _graph_state["nodes"] = [
        {"operationId": "createPet"},
        {"operationId": "getPetById"}
    ]
    _graph_state["edges"] = [
        {"from": "createPet", "to": "getPetById"}
    ]
    return _graph_state

@tool
def add_edge_fn(user_instruction: str) -> dict:
    """Use LLM to interpret and add an edge to the graph, with validation."""
    if "from" in user_instruction and "to" in user_instruction:
        tokens = user_instruction.lower().replace("'", "").split()
        src = tokens[tokens.index("from") + 1]
        tgt = tokens[tokens.index("to") + 1]

        existing_ids = {n["operationId"] for n in _graph_state["nodes"]}
        if src not in existing_ids:
            _graph_state["nodes"].append({"operationId": src})
        if tgt not in existing_ids:
            _graph_state["nodes"].append({"operationId": tgt})

        if not any(e for e in _graph_state["edges"] if e["from"] == src and e["to"] == tgt):
            _graph_state["edges"].append({"from": src, "to": tgt})

    return _graph_state

@tool
def describe_execution_plan_fn(_: str = "") -> str:
    """Return a textual description of the current execution graph."""
    steps = []
    for edge in _graph_state["edges"]:
        steps.append(f"Call '{edge['from']}' before '{edge['to']}'")
    return "Execution Plan:\n" + "\n".join(steps) if steps else "No execution steps defined."

@tool
def get_execution_graph_json_fn(_: str = "") -> dict:
    """Return the full graph with nodes and edges as JSON."""
    return _graph_state

@tool
def validate_graph_fn(graph: dict) -> str:
    """Validate the execution graph (e.g., no unreachable nodes)."""
    if not graph.get("nodes") or not graph.get("edges"):
        return "Graph is empty or incomplete."

    op_ids = {n["operationId"] for n in graph["nodes"]}
    from_ops = {e["from"] for e in graph["edges"]}
    to_ops = {e["to"] for e in graph["edges"]}

    unreachable = op_ids - from_ops - to_ops
    if unreachable:
        return f"Warning: Unreachable nodes found: {unreachable}"

    return "Graph is valid."
