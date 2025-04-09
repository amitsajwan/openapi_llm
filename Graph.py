from langgraph.graph import StateGraph
from typing import TypedDict
import httpx
import asyncio
import re

# --------------------
# State Definition
# --------------------
class APIState(TypedDict, total=False):
    variables: dict
    results: dict
    confirmed: dict
    payloads: dict
    first_run: bool
    step: str
    action: str
    target_node: str
    payload: dict


# --------------------
# Payload Generator (Replace with yours)
# --------------------
def generate_payload_fn(method: str, endpoint: str) -> dict:
    # Example dummy payload based on method/endpoint
    if method.lower() == "post" and "pet" in endpoint:
        return {"name": "Doggie", "type": "dog"}
    if method.lower() == "put" and "pet" in endpoint:
        return {"name": "DoggieUpdated", "type": "dog"}
    return {}


# --------------------
# Actual API Caller
# --------------------
async def api_call_fn(method: str, endpoint: str, payload: dict = None) -> dict:
    base_url = "https://petstore3.swagger.io/api/v3"
    url = base_url + endpoint

    async with httpx.AsyncClient() as client:
        response = await client.request(method.upper(), url, json=payload)
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}


# --------------------
# Extract variables (IDs, etc.)
# --------------------
def extract_variables_from_response(response: dict) -> dict:
    vars_found = {}

    # Common patterns
    if "id" in response:
        vars_found["id"] = response["id"]

    # Nested extraction
    if isinstance(response, dict):
        for key, value in response.items():
            if isinstance(value, dict) and "id" in value:
                vars_found[key + "_id"] = value["id"]
    return vars_found


# --------------------
# Terminal nodes finder
# --------------------

# 
def find_terminal_nodes(graph: dict) -> list:
    all_nodes = {node["name"] for node in graph["nodes"]}
    from_nodes = {e["from"] for e in graph["edges"]}
    return list(all_nodes - from_nodes)  # nodes that have no outgoing edges
------------------
# Dynamic LangGraph Builder
# --------------------
def build_api_graph(graph_json: dict) -> StateGraph:
    sg = StateGraph(APIState)
    node_map = {}

    for node in graph_json["nodes"]:
        node_name = node["name"]
        method = node["method"].lower()
        endpoint = node["endpoint"]

        async def node_fn(state: APIState, node=node):
            name = node["name"]
            method = node["method"].lower()
            endpoint = node["endpoint"]

            payload = None
            if method in ("post", "put"):
                payload = generate_payload_fn(method, endpoint)
                if state.get("first_run", True) and not state.get("confirmed", {}).get(name):
                    return {
                        "action": "confirm_payload",
                        "payload": payload,
                        "target_node": name,
                        "step": name
                    }

                payload = state.get("payloads", {}).get(name, payload)

            # Format endpoint
            variables = state.get("variables", {})
            try:
                formatted_endpoint = endpoint.format(**variables)
            except KeyError:
                formatted_endpoint = endpoint

            response = await api_call_fn(method, formatted_endpoint, payload)
            extracted = extract_variables_from_response(response)

            return {
                "step": name,
                "results": {name: response},
                "variables": {**variables, **extracted}
            }

        sg.add_node(node_name, node_fn)
        node_map[node_name] = node_fn

    # Add edges
    for edge in graph_json["edges"]:
        sg.add_edge(edge["from"], edge["to"])

    # Entry and finish
    entry_node = graph_json["nodes"][0]["name"]
    finish_nodes = find_terminal_nodes(graph_json)

    sg.set_entry_point(entry_node)
    sg.set_finish_condition(lambda state: state.get("step") in finish_nodes)

    return sg.compile()
