import json
from typing import Any, Dict
from langchain_core.tools import tool
import networkx as nx

# In-memory graph state
_graph_state = {"nodes": [], "edges": []}


@tool
def general_query_fn(query: str, llm) -> Dict[str, Any]:
    """Answer any general API-related question using LLM reasoning."""
    try:
        response = llm.invoke(f"Answer this query: {query}")
        return {
            "intent": "general_query",
            "user_input": query,
            "response": response.content
        }
    except Exception as e:
        return {
            "intent": "general_query",
            "user_input": query,
            "response": f"Error processing query: {str(e)}"
        }


@tool
def openapi_help_fn(question: str, openapi_yaml: str, llm) -> Dict[str, Any]:
    """Explain what an OpenAPI endpoint does given the spec."""
    try:
        prompt = f"Explain what the following OpenAPI spec describes:\n\n{openapi_yaml}\n\nQuestion: {question}"
        response = llm.invoke(prompt)
        return {
            "intent": "openapi_help",
            "user_input": question,
            "response": response.content
        }
    except Exception as e:
        return {
            "intent": "openapi_help",
            "user_input": question,
            "response": f"Error explaining OpenAPI spec: {str(e)}"
        }


@tool
def generate_payload_fn(endpoint: str, schema: dict, llm) -> Dict[str, Any]:
    """Generate a realistic JSON payload matching the given schema."""
    try:
        prompt = f"Generate a realistic JSON payload for the endpoint '{endpoint}' using this schema: {schema}"
        response = llm.invoke(prompt)
        return {
            "intent": "generate_payload",
            "user_input": endpoint,
            "response": json.loads(response.content)
        }
    except Exception as e:
        return {
            "intent": "generate_payload",
            "user_input": endpoint,
            "response": {"error": str(e), "message": "Failed to generate payload"}
        }


@tool
def generate_api_execution_graph_fn(user_input: str, spec_text: str, llm) -> Dict[str, Any]:
    """Generates an API execution graph (nodes & edges) from OpenAPI spec and user input."""
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
        """
        response = llm.invoke(prompt_template)
        return {
            "intent": "generate_api_execution_graph",
            "user_input": user_input,
            "response": json.loads(response.content)
        }
    except Exception as e:
        return {
            "intent": "generate_api_execution_graph",
            "user_input": user_input,
            "response": {"error": str(e), "message": "Failed to generate execution graph"}
        }


@tool
def add_edge_fn(user_instruction: str, llm) -> Dict[str, Any]:
    """Add an edge to the graph based on user instruction."""
    try:
        prompt = f"""
        Interpret this instruction: '{user_instruction}'.
        Current graph state: {json.dumps(_graph_state, indent=2)}

        Add an edge to the graph based on the instruction and return the updated graph.
        """
        response = llm.invoke(prompt)
        updated_graph = json.loads(response.content)
        _graph_state.update(updated_graph)  # Update the in-memory graph state
        return {
            "intent": "add_edge",
            "user_input": user_instruction,
            "response": updated_graph
        }
    except Exception as e:
        return {
            "intent": "add_edge",
            "user_input": user_instruction,
            "response": {"error": str(e), "message": "Failed to add edge"}
        }


@tool
def describe_execution_plan_fn(graph: dict, llm) -> Dict[str, Any]:
    """Describe the planned execution of the API calls based on the graph."""
    try:
        prompt = f"""
        Given this API execution graph, explain the sequence of execution in natural language:
        {json.dumps(graph, indent=2)}
        """
        response = llm.invoke(prompt)
        return {
            "intent": "describe_execution_plan",
            "user_input": "",
            "response": response.content
        }
    except Exception as e:
        return {
            "intent": "describe_execution_plan",
            "user_input": "",
            "response": f"Error describing execution plan: {str(e)}"
        }


@tool
def validate_graph_fn(graph: dict) -> Dict[str, Any]:
    """Validate the execution graph (e.g., no unreachable nodes, no cycles)."""
    try:
        G = nx.DiGraph()
        for node in graph.get("nodes", []):
            G.add_node(node["operationId"])
        for edge in graph.get("edges", []):
            G.add_edge(edge["from"], edge["to"])

        if not nx.is_directed_acyclic_graph(G):
            return {
                "intent": "validate_graph",
                "user_input": "",
                "response": "Graph contains cycles."
            }

        if not nx.is_weakly_connected(G):
            return {
                "intent": "validate_graph",
                "user_input": "",
                "response": "Graph is disconnected."
            }

        return {
            "intent": "validate_graph",
            "user_input": "",
            "response": "Graph is valid."
        }
    except Exception as e:
        return {
            "intent": "validate_graph",
            "user_input": "",
            "response": f"Error validating graph: {str(e)}"
        }


@tool
def get_execution_graph_json_fn(_: str = "") -> Dict[str, Any]:
    """Return the full graph with nodes and edges as JSON."""
    try:
        return {
            "intent": "get_execution_graph_json",
            "user_input": "",
            "response": _graph_state
        }
    except Exception as e:
        return {
            "intent": "get_execution_graph_json",
            "user_input": "",
            "response": f"Error retrieving graph JSON: {str(e)}"
        }


@tool
def execute_workflow_fn(_: str, llm) -> Dict[str, Any]:
    """Execute workflow based on the current graph JSON."""
    try:
        # Step 1: Retrieve the current graph JSON
        graph_json = get_execution_graph_json_fn(_: "").get("response")

        if not graph_json or not graph_json.get("nodes") or not graph_json.get("edges"):
            return {
                "intent": "execute_workflow",
                "user_input": "",
                "response": "Error: Graph JSON is empty or incomplete. Cannot execute workflow."
            }

        # Step 2: Simulate workflow execution
        prompt = f"""
        Based on the following API execution graph, simulate executing the workflow step by step:
        {json.dumps(graph_json, indent=2)}

        Provide the output of each step in a concise format.
        """
        response = llm.invoke(prompt)

        return {
            "intent": "execute_workflow",
            "user_input": "",
            "response": response.content
        }
    except Exception as e:
        return {
            "intent": "execute_workflow",
            "user_input": "",
            "response": f"Error executing workflow: {str(e)}"
        }
