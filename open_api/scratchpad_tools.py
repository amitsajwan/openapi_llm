import os, json
from typing import Dict, Any
from models import RouterState
from utils.openapi_parser import OpenAPIParser  # your parser
from utils.sequence_generator import suggest_execution_order
from utils.payload_generator import PayloadGenerator
from executor.api_executor import APIExecutor
from utils.graph_utils import validate_execution_graph, add_edge_to_graph

def parse_openapi(state: RouterState) -> RouterState:
    if 'openapi_schema' not in state.scratchpad:
        parser = OpenAPIParser()
        schema = parser.parse_from_text(state.openapi_schema or "")
        state.scratchpad['openapi_schema'] = schema
    state.openapi_schema = state.scratchpad['openapi_schema']
    state.response = "Parsed OpenAPI schema."
    return state

def generate_sequence(state: RouterState) -> RouterState:
    if not state.openapi_schema:
        return state  # require parse_openapi first
    graph = suggest_execution_order(state.openapi_schema)
    state.execution_graph = graph
    state.response = "Generated execution graph."
    state.scratchpad['execution_graph'] = graph
    return state

def generate_payloads(state: RouterState) -> RouterState:
    if 'payloads' not in state.scratchpad:
        gen = PayloadGenerator(state.openapi_schema)
        p = gen.generate_all()
        state.scratchpad['payloads'] = p
    state.payloads = state.scratchpad['payloads']
    state.response = "Generated payloads."
    return state

def answer(state: RouterState) -> RouterState:
    state.response = f"Answer: {state.user_input}"
    return state

def simulate_load_test(state: RouterState) -> RouterState:
    users = state.scratchpad.get('last_load_users', 10)
    executor = APIExecutor(state.openapi_schema, state.execution_graph, state.payloads,
                           simulate_load=True, concurrent_users=users)
    result = executor.run()
    state.response = "Load test simulated."
    state.scratchpad['last_load_result'] = result
    return state

def execute_workflow(state: RouterState) -> RouterState:
    executor = APIExecutor(state.openapi_schema, state.execution_graph, state.payloads)
    result = executor.run()
    state.response = "Workflow executed."
    state.scratchpad['last_run_result'] = result
    return state

def add_edge(state: RouterState) -> RouterState:
    src, tgt = state.user_input.split("->")
    graph = state.execution_graph or {}
    new_graph = add_edge_to_graph(graph, src.strip(), tgt.strip())
    state.execution_graph = new_graph
    state.response = "Edge added."
    return state

def validate_graph(state: RouterState) -> RouterState:
    valid, msg = validate_execution_graph(state.execution_graph or {})
    state.response = msg
    return state

def unknown_intent(state: RouterState) -> RouterState:
    state.response = "Sorry, didn't understand."
    return state


def list_apis(state: RouterState) -> RouterState:
    """
    List all available API operations (method + path) from the OpenAPI spec.
    """
    spec = state.openapi_schema or {}
    apis = []
    for path, methods in spec.get("paths", {}).items():
        for method in methods:
            apis.append(f"{method.upper()} {path}")
    state.response = "Available APIs:\n" + "\n".join(apis)
    state.scratchpad['last_list_apis'] = apis
    return state


def explain_endpoint(state: RouterState) -> RouterState:
    """
    Explain details of a specific endpoint named or given in user_input.
    """
    user = state.user_input or ""
    # attempt to parse method and path
    parts = user.split()
    if len(parts) >= 2 and parts[0].upper() in ["GET","POST","PUT","DELETE","PATCH"]:
        method, path = parts[0].upper(), parts[1]
    else:
        state.response = "Please specify endpoint as '<METHOD> <path>'."
        return state

    details = (state.openapi_schema or {}).get("paths", {}).get(path, {}).get(method.lower())
    if not details:
        state.response = f"Endpoint {method} {path} not found."
        return state

    desc = details.get("description", "No description.")
    params = details.get("parameters", [])
    rb = details.get("requestBody", {})
    state.response = (
        f"{method} {path}: {desc}\n"
        f"Parameters: {json.dumps(params, indent=2)}\n"
        f"RequestBody schema: {json.dumps(rb.get('content', {}), indent=2)}"
    )
    return state


def explain_graph(state: RouterState) -> RouterState:
    """
    Provide a human-readable explanation of the current execution graph.
    """
    graph = state.execution_graph or {}
    lines = []
    for node, data in graph.items():
        nxt = data.get("next", [])
        lines.append(f"{node} -> {', '.join(nxt) if nxt else 'END'}")
    state.response = "Execution Graph:\n" + "\n".join(lines)
    return state


def verify_created(state: RouterState) -> RouterState:
    """
    After a create/update step, verify by fetching the resource.
    Assumes last payload contained an 'id' field.
    """
    payloads = state.payloads or {}
    last = state.scratchpad.get('last_payload_op')
    if not last:
        state.response = "No record of a creation operation to verify."
        return state
    op_id, payload = last
    resource_id = payload.get('id')
    if not resource_id:
        state.response = "No 'id' found in payload to verify."
        return state
    # assume GET operationId is 'get' + resource
    get_op = f"get{op_id[0].upper()}{op_id[1:]}"
    if get_op in payloads:
        state.response = f"Verified resource via {get_op} with id {resource_id}."
    else:
        state.response = f"Could not verify: GET operation {get_op} not found."
    return state


 
