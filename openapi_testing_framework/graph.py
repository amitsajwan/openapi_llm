"""
graph.py

Defines the LangGraph StateGraph for the OpenAPI testing framework.
Nodes:
  - router: determines next action
  - load_spec, list_apis, generate_sequence, generate_payload, call_api, validate_graph, save_results
Edges are dynamic: router -> next tool -> router until plan empty.

Integrates MemorySaver for state persistence.
"""

from langgraph import StateGraph, MemorySaver
from router import router
from tools import (
    load_spec, list_apis, generate_sequence,
    generate_payload, call_api, validate_graph, save_results
)


def build_openapi_graph() -> StateGraph:
    # Create a new graph
    graph = StateGraph(name="openapi_test_flow")

    # Add router node
    graph.add_node(router, name="router")

    # Add tool nodes
    graph.add_node(load_spec, name="load_spec")
    graph.add_node(list_apis, name="list_apis")
    graph.add_node(generate_sequence, name="generate_sequence")
    graph.add_node(generate_payload, name="generate_payload")
    graph.add_node(call_api, name="call_api")
    graph.add_node(validate_graph, name="validate_graph")
    graph.add_node(save_results, name="save_results")

    # Dynamic edges: router outputs next_action -> that tool -> back to router
    # LangGraph will route based on router.next_action, so we only need a generic feedback edge
    graph.add_edge("router", "load_spec")
    graph.add_edge("router", "list_apis")
    graph.add_edge("router", "generate_sequence")
    graph.add_edge("router", "generate_payload")
    graph.add_edge("router", "call_api")
    graph.add_edge("router", "validate_graph")
    graph.add_edge("router", "save_results")
    graph.add_edge("load_spec", "router")
    graph.add_edge("list_apis", "router")
    graph.add_edge("generate_sequence", "router")
    graph.add_edge("generate_payload", "router")
    graph.add_edge("call_api", "router")
    graph.add_edge("validate_graph", "router")
    graph.add_edge("save_results", "router")

    # Attach MemorySaver for checkpointing plan and results
    memory = MemorySaver()
    graph.use_saver(memory)

    return graph
