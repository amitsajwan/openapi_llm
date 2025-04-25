from langchain_core.tools import tool
from typing import Any

# In-memory graph state
_graph_state = {"nodes": [], "edges": []}

from langchain_core.tools import tool
from typing import Any

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
    return {"example": f"Payload for {endpoint}"}

@tool
def generate_api_execution_graph_fn(openapi_yaml: str) -> dict:
    """Produce a dependency-aware execution graph (nodes & edges) from OpenAPI spec."""
    # This would use LLM and parsed YAML spec to identify endpoints and order.
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
    # Simulate interpretation using simple parsing (LLM can be added)
    if "from" in user_instruction and "to" in user_instruction:
        tokens = user_instruction.lower().replace("'", "").split()
        src = tokens[tokens.index("from") + 1]
        tgt = tokens[tokens.index("to") + 1]
        
        existing_ids = {n["operationId"] for n in _graph_state["nodes"]}
        if src not in existing_ids:
            _graph_state["nodes"].append({"operationId": src})
        if tgt not in existing_ids:
            _graph_state["nodes"].append({"operationId": tgt})

        # Validate if edge exists
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


# File: agent.py
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import ChatOpenAI
from tools import (
    general_query_fn,
    openapi_help_fn,
    generate_payload_fn,
    generate_api_execution_graph_fn,
    add_edge_fn,
    describe_execution_plan_fn,
    get_execution_graph_json_fn,
)



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
    return {"example": f"Payload for {endpoint}"}

@tool
def generate_api_execution_graph_fn(user_input: str, spec_text: str) -> dict:
    """
    Generates an API execution graph (nodes & edges) from OpenAPI spec and user input.
    """
    prompt_template = f"""
You are an expert API testing assistant. Use the following OpenAPI spec:

{spec_text}

{user_input}

Generate an API execution graph in JSON format with 'nodes' and 'edges'.

Follow these rules:
1. POST must come before GET/PUT/DELETE for the same resource.
2. Add 'verify' nodes after every POST/PUT.
3. Only parallel execution if operations are independent.
4. Use operationId from the OpenAPI spec as node identifiers.
5. Resolve $ref, enums, and nested schemas for realistic payloads.

Return output like:
{{
  "nodes": [
    {{
      "operationId": "createPet",
      "method": "POST",
      "path": "/pets",
      "description": "Create a new pet",
      "payload": {{ "name": "Fido", "type": "dog" }}
    }},
    ...
  ],
  "edges": [
    {{ "from_node": "createPet", "to_node": "verifyCreatePet" }},
    ...
  ]
}}
"""
    response = llm.invoke(prompt_template)
    return {
        "response": response,
        "intent": "execute workflow",
        "query": user_input
    }


@tool
def add_edge_fn(user_instruction: str) -> dict:
    """Use LLM to interpret and add an edge to the graph, with validation."""
    # Simulate interpretation using simple parsing (LLM can be added)
    if "from" in user_instruction and "to" in user_instruction:
        tokens = user_instruction.lower().replace("'", "").split()
        src = tokens[tokens.index("from") + 1]
        tgt = tokens[tokens.index("to") + 1]
        
        existing_ids = {n["operationId"] for n in _graph_state["nodes"]}
        if src not in existing_ids:
            _graph_state["nodes"].append({"operationId": src})
        if tgt not in existing_ids:
            _graph_state["nodes"].append({"operationId": tgt})

        # Validate if edge exists
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
def describe_execution_plan_fn(graph: dict) -> str:
    """Generate natural language description of the current execution graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    steps = [f"- {node.get('operationId')}" for node in nodes]
    desc = "\n".join(steps)
    return f"The workflow includes the following steps:\n{desc}"

@tool
def validate_graph_fn(graph: dict) -> str:
    """Validate the execution graph (e.g., no unreachable nodes)."""
    if not graph.get("nodes") or not graph.get("edges"):
        return "Graph is empty or incomplete."

    # Just a basic check for example
    op_ids = {n["operationId"] for n in graph["nodes"]}
    from_ops = {e["from"] for e in graph["edges"]}
    to_ops = {e["to"] for e in graph["edges"]}

    unreachable = op_ids - from_ops - to_ops
    if unreachable:
        return f"Warning: Unreachable nodes found: {unreachable}"

    return "Graph is valid."

@tool
def get_graph_json_fn(graph: dict) -> dict:
    """Return the full graph structure as JSON."""
    return graph



