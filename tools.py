import json
from typing import Any, List
from langchain_core.tools import tool
import networkx as nx

# In-memory graph state
_graph_state = {"nodes": [], "edges": []}

@tool
def general_query_fn(query: str) -> str:
    """Answer any general API-related question using LLM reasoning."""
    try:
        response = llm.invoke(f"Answer this query: {query}")
        return response.content
    except Exception as e:
        return f"Error processing query: {str(e)}"

@tool
def openapi_help_fn(question: str, openapi_yaml: str) -> str:
    """Explain what an OpenAPI endpoint does given the spec."""
    try:
        prompt = f"Explain what the following OpenAPI spec describes:\n\n{openapi_yaml}\n\nQuestion: {question}"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error explaining OpenAPI spec: {str(e)}"

@tool
def generate_payload_fn(endpoint: str, schema: dict) -> dict:
    """Generate a realistic JSON payload matching the given schema."""
    try:
        prompt = f"Generate a realistic JSON payload for the endpoint '{endpoint}' using this schema: {schema}"
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        return {"error": str(e), "message": "Failed to generate payload"}

@tool
def generate_api_execution_graph_fn(user_input: str, spec_text: str) -> dict:
    """
    Generates an API execution graph (nodes & edges) from OpenAPI spec and user input.
    """
    try:
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
        return json.loads(response.content)
    except Exception as e:
        return {"error": str(e), "message": "Failed to generate execution graph"}

@tool
def add_edge_fn(user_instruction: str) -> dict:
    """Use LLM to interpret and add an edge to the graph, with validation."""
    try:
        prompt = f"Interpret and add an edge to the graph based on this instruction: '{user_instruction}'. Current graph state: {_graph_state}"
        response = llm.invoke(prompt)
        updated_graph = json.loads(response.content)
        return updated_graph
    except Exception as e:
        return {"error": str(e), "message": "Failed to add edge"}

@tool
def describe_execution_plan_fn(graph: dict) -> str:
    """Describe the planned execution of the API calls based on the graph."""
    try:
        prompt = f"""
        Given this API execution graph (nodes and edges), explain the sequence of execution
        in natural language. Be concise but clear. Mention the order, any dependencies,
        and describe what each step does.

        Graph:
        {json.dumps(graph, indent=2)}
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error describing execution plan: {str(e)}"

@tool
def validate_graph_fn(graph: dict) -> str:
    """Validate the execution graph (e.g., no unreachable nodes)."""
    try:
        G = nx.DiGraph()
        for node in graph.get("nodes", []):
            G.add_node(node["operationId"])
        for edge in graph.get("edges", []):
            G.add_edge(edge["from"], edge["to"])

        if not nx.is_directed_acyclic_graph(G):
            return "Graph contains cycles."
        if not nx.is_weakly_connected(G):
            return "Graph is disconnected."

        unreachable = [n for n in G.nodes if nx.in_degree(G, n) == 0]
        if unreachable:
            return f"Warning: Unreachable nodes found: {unreachable}"

        return "Graph is valid."
    except Exception as e:
        return f"Error validating graph: {str(e)}"

@tool
def get_execution_graph_json_fn(_: str = "") -> dict:
    """Return the full graph with nodes and edges as JSON."""
    return _graph_state
