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
def execute_workflow_fn(_: str, llm) -> Dict[str, Any]:
    """Execute workflow based on the current graph JSON."""
    try:
        # Step 1: Retrieve the current graph JSON
        graph_json = get_execution_graph_json_fn()

        # Step 2: Simulate workflow execution
        # Replace this with your actual workflow execution logic
        prompt = f"""
        Based on the following API execution graph, simulate executing the workflow step by step:
        {json.dumps(graph_json, indent=2)}

        Provide the output of each step in a concise format.
        """
        response = llm.invoke(prompt)

        # Step 3: Return the result
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
