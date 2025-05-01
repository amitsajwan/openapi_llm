# graph.py

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from models import BotState
from router import router
from tools import (
    load_spec,
    list_apis,
    generate_sequence,
    generate_payload,
    call_api,
    validate_graph,
    save_results,
)

def build_openapi_graph():
    # 1) Initialize StateGraph with the BotState schema
    graph = StateGraph(state_schema=BotState, name="openapi_test_flow")  

    # 2) Add the router node as the entry point
    graph.add_node("router", router)  

    # 3) Add each tool node correctly: (key, function)
    graph.add_node("load_spec", load_spec)
    graph.add_node("list_apis", list_apis)
    graph.add_node("generate_sequence", generate_sequence)
    graph.add_node("generate_payload", generate_payload)
    graph.add_node("call_api", call_api)
    graph.add_node("validate_graph", validate_graph)
    graph.add_node("save_results", save_results)

    # 4) Wire edges: router → tool → router
    tool_names = [
        "load_spec", "list_apis", "generate_sequence",
        "generate_payload", "call_api", "validate_graph", "save_results"
    ]
    for t in tool_names:
        graph.add_edge("router", t)
        graph.add_edge(t, "router")

    # 5) Set the entry point so execution begins at the router
    graph.set_entry_point("router")  

    # 6) (Optional) You can set a finish point, or rely on edges to END
    # graph.set_finish_point("save_results")

    # 7) Attach MemorySaver for checkpointing state
    memory = MemorySaver()
    graph.use_saver(memory)

    # 8) Compile into a runnable graph (supports .invoke, .stream, .astream)
    compiled = graph.compile(checkpointer=memory)
    return compiled
