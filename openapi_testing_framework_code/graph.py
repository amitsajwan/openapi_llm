from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from router import router
from tools import load_spec, list_apis, generate_sequence, generate_payload, call_api, validate_graph, save_results

def build_openapi_graph():
    graph = StateGraph(name="openapi_test_flow")
    graph.add_node(router, name="router")
    nodes = [
        (load_spec, "load_spec"),
        (list_apis, "list_apis"),
        (generate_sequence, "generate_sequence"),
        (generate_payload, "generate_payload"),
        (call_api, "call_api"),
        (validate_graph, "validate_graph"),
        (save_results, "save_results"),
    ]
    for fn, nm in nodes:
        graph.add_node(fn, name=nm)
        graph.add_edge("router", nm)
        graph.add_edge(nm, "router")
    memory = MemorySaver()
    graph.use_saver(memory)
    return graph.compile(checkpointer=memory)
