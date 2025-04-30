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

TOOL_FUNCTIONS = {
    "parse_openapi": parse_openapi,
    "generate_sequence": generate_sequence,
    "generate_payloads": generate_payloads,
    "answer": answer,
    "simulate_load_test": simulate_load_test,
    "execute_workflow": execute_workflow,
    "add_edge": add_edge,
    "validate_graph": validate_graph,
    "unknown_intent": unknown_intent,
}

