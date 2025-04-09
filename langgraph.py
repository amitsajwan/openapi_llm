from langgraph.graph import StateGraph
from typing import TypedDict
import asyncio

class APIState(TypedDict, total=False):
    variables: dict
    results: dict
    confirmed: dict
    payloads: dict
    first_run: bool

def build_api_graph(graph_json: dict, generate_payload_fn, api_call_fn):
    sg = StateGraph(APIState)

    node_map = {}

    for node in graph_json["nodes"]:
        node_name = node["name"]
        method = node["method"].lower()
        endpoint = node["endpoint"]

        async def node_fn(state: APIState, node=node):
            name = node["name"]
            method = node["method"]
            endpoint = node["endpoint"]

            payload = None
            if method in ("post", "put"):
                payload = generate_payload_fn(method, endpoint)

                if state.get("first_run") and not state.get("confirmed", {}).get(name):
                    return {
                        "action": "confirm_payload",
                        "payload": payload,
                        "target_node": name
                    }

                # Use confirmed payload if it exists
                payload = state.get("payloads", {}).get(name, payload)

            # Format endpoint with variables
            formatted_endpoint = endpoint.format(**state.get("variables", {}))
            response = await api_call_fn(method, formatted_endpoint, payload)

            # Optionally extract IDs or tokens
            new_vars = extract_variables_from_response(response)
            return {
                "results": {name: response},
                "variables": {**state.get("variables", {}), **new_vars}
            }

        sg.add_node(node_name, node_fn)
        node_map[node_name] = node_fn

    # Add edges
    for edge in graph_json["edges"]:
        sg.add_edge(edge["from"], edge["to"])

    # Define input/output
    first_node = graph_json["nodes"][0]["name"]
    last_nodes = find_terminal_nodes(graph_json)

    sg.set_entry_point(first_node)
    sg.set_finish_condition(lambda state: state.get("step") in last_nodes)

    return sg.compile()
