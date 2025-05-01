import os
import json
import logging
from typing import Any, Dict, List
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from models import (
    BotState, GraphOutput, Edge, TextContent, BotIntent,
    AddEdgeParams, GeneratePayloadParams
)

# Module-level logger
logger = logging.getLogger(__name__)

# Directory for caching resolved schemas
CACHE_DIR = os.getenv("OPENAPI_CACHE_DIR", "./schema_cache")
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception as e:
    logger.warning(f"Could not create cache directory {CACHE_DIR}: {e}")

# Initialize MemorySaver checkpointer
memory_checkpointer = MemorySaver()

# Helper for robust LLM calls with retry and error handling
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def llm_call_helper(llm: Any, prompt: str, **kwargs) -> str:
    """
    Invoke the LLM with retries and exponential backoff on transient errors.
    Returns the raw LLM response string or raises after maximum retries.
    """
    try:
        response = llm.invoke(prompt, **kwargs)
        return response
    except Exception as e:
        logger.error(f"LLM invocation error: {e}", exc_info=True)
        raise

# --- Helper for programmatic cycle detection ---
def check_for_cycles(graph: GraphOutput) -> tuple[bool, str]:
    """Programmatically check for cycles in the graph using DFS."""
    nodes = {node.operationId for node in graph.nodes}
    adj: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    for edge in graph.edges:
        if edge.from_node in adj and edge.to_node in adj:
            adj[edge.from_node].append(edge.to_node)

    visited: Dict[str, int] = {}
    recursion_stack: Dict[str, bool] = {}

    def dfs(node_id: str) -> bool:
        visited[node_id] = 1
        recursion_stack[node_id] = True
        for neighbor in adj.get(node_id, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif recursion_stack.get(neighbor, False):
                return True
        recursion_stack[node_id] = False
        visited[node_id] = 2
        return False

    for node_id in nodes:
        if node_id not in visited:
            if dfs(node_id):
                return False, "Cycle detected in the graph."
    return True, "No cycles detected."

# --- Scratchpad helper ---
def update_scratchpad_reason(state: BotState, tool_name: str, details: str) -> BotState:
    """Helper to append reasoning/details to the scratchpad."""
    current = state.scratchpad.get('reason', '')
    new_entry = f"\nTool: {tool_name}\nDetails: {details}\n"
    combined = (current + new_entry)[-2000:]
    state.scratchpad['reason'] = combined
    return state

# --- Tool Functions ---
def parse_openapi(state: BotState, llm: Any) -> BotState:
    """Tool to parse OpenAPI spec text into a schema."""
    tool_name = BotIntent.OPENAPI_HELP.value
    if not state.openapi_spec_text:
        state.text_content = TextContent(text="Error: No OpenAPI spec text provided to parse.")
        return update_scratchpad_reason(state, tool_name, "No spec text provided.")

    prompt = f"Parse this OpenAPI spec into a JSON schema object. Output only the JSON object.\nSpec:\n{state.openapi_spec_text}"
    try:
        res = llm_call_helper(llm, prompt)
        schema = json.loads(res)
        state.openapi_schema = schema
        state.text_content = TextContent(text="OpenAPI schema parsed successfully.")
        return update_scratchpad_reason(state, tool_name, f"Parsed schema keys: {list(schema.keys())}")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error parsing OpenAPI schema: {e}")
        return update_scratchpad_reason(state, tool_name, f"Parsing failed: {e}")

def generate_api_execution_graph(state: BotState, llm: Any) -> BotState:
    """Tool to generate an execution graph from the parsed schema."""
    tool_name = BotIntent.GENERATE_GRAPH.value
    if not state.openapi_schema:
        state.text_content = TextContent(text="Error: OpenAPI schema not available. Parse spec first.")
        return update_scratchpad_reason(state, tool_name, "Schema missing.")

    prompt = (
        f"Generate a DAG as GraphOutput JSON from this OpenAPI schema. Schema:\n{json.dumps(state.openapi_schema)}"
    )
    try:
        res = llm_call_helper(llm, prompt)
        graph_data = json.loads(res)
        state.graph_output = GraphOutput(**graph_data)
        state.text_content = TextContent(text="Execution graph generated successfully.")
        return update_scratchpad_reason(state, tool_name, f"Nodes: {len(state.graph_output.nodes)}, Edges: {len(state.graph_output.edges)}")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error generating graph: {e}")
        state.graph_output = None
        return update_scratchpad_reason(state, tool_name, f"Graph gen failed: {e}")

def generate_payload(state: BotState, llm: Any) -> BotState:
    """Tool to generate example payloads and update graph nodes."""
    tool_name = BotIntent.GENERATE_PAYLOAD.value
    if not state.openapi_schema or not state.graph_output:
        state.text_content = TextContent(text="Error: Schema or graph missing for payload generation.")
        return update_scratchpad_reason(state, tool_name, "Prerequisites missing.")

    prompt = (
        f"Generate example payloads for graph nodes. Schema:\n{json.dumps(state.openapi_schema)}\nGraph:\n{state.graph_output.model_dump_json()}"
    )
    try:
        res = llm_call_helper(llm, prompt)
        graph_data = json.loads(res)
        state.graph_output = GraphOutput(**graph_data)
        state.text_content = TextContent(text="Payloads generated and added.")
        return update_scratchpad_reason(state, tool_name, "Payloads updated.")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error generating payloads: {e}")
        return update_scratchpad_reason(state, tool_name, f"Payload gen failed: {e}")

def add_edge(state: BotState, llm: Any) -> BotState:
    """Tool to modify the execution graph by adding an edge."""
    tool_name = BotIntent.ADD_EDGE.value
    if not state.graph_output or not state.extracted_params:
        state.text_content = TextContent(text="Error: Graph or parameters missing for add_edge.")
        return update_scratchpad_reason(state, tool_name, "Prerequisites missing.")
    try:
        params = AddEdgeParams(**state.extracted_params)
        nodes_dict = {n.operationId: n for n in state.graph_output.nodes}
        if params.from_node not in nodes_dict or params.to_node not in nodes_dict:
            state.text_content = TextContent(text="Error: Specified nodes not found.")
            return update_scratchpad_reason(state, tool_name, "Node lookup failed.")
        new_edge = Edge(from_node=params.from_node, to_node=params.to_node)
        if new_edge in state.graph_output.edges:
            state.text_content = TextContent(text="Edge already exists.")
            return update_scratchpad_reason(state, tool_name, "No-op, edge exists.")
        updated = state.graph_output.copy()
        updated.edges.append(new_edge)
        valid, reason = check_for_cycles(updated)
        if not valid:
            state.text_content = TextContent(text=f"Error: Adding edge creates cycle. {reason}")
            return update_scratchpad_reason(state, tool_name, f"Cycle detected: {reason}")
        state.graph_output = updated
        state.text_content = TextContent(text="Edge added successfully.")
        return update_scratchpad_reason(state, tool_name, f"Added edge {params.from_node}->{params.to_node}.")
    except ValidationError as e:
        state.text_content = TextContent(text=f"Parameter validation error: {e}")
        return update_scratchpad_reason(state, tool_name, f"Param validation failed: {e}")

def validate_graph(state: BotState, llm: Any) -> BotState:
    """Tool to validate the current execution graph (cycle check + user explanation)."""
    tool_name = BotIntent.VALIDATE_GRAPH.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No graph to validate.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")
    valid, reason = check_for_cycles(state.graph_output)
    prompt = f"Explain this graph validation: Valid={valid}, Reason={reason}."
    res = llm_call_helper(llm, prompt)
    state.text_content = TextContent(text=f"Validation: {valid}. {res}")
    return update_scratchpad_reason(state, tool_name, f"Validation done: {valid}")

def describe_execution_plan(state: BotState, llm: Any) -> BotState:
    """Tool to generate a natural-language description of the execution graph."""
    tool_name = BotIntent.DESCRIBE_GRAPH.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No graph to describe.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")
    prompt = f"Describe this execution plan: {state.graph_output.model_dump_json()}"
    res = llm_call_helper(llm, prompt)
    state.text_content = TextContent(text=res)
    return update_scratchpad_reason(state, tool_name, f"Described plan, length {len(res)}.")

def get_execution_graph_json(state: BotState, llm: Any) -> BotState:
    """Tool to output the current execution graph JSON."""
    tool_name = BotIntent.GET_GRAPH_JSON.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No graph to output.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")
    state.text_content = TextContent(text=state.graph_output.model_dump_json(indent=2))
    return update_scratchpad_reason(state, tool_name, "Graph JSON output.")

def openapi_help(state: BotState, llm: Any) -> BotState:
    """Tool for help on OpenAPI or initial spec parsing."""
    tool_name = BotIntent.OPENAPI_HELP.value
    if state.openapi_spec_text and not state.openapi_schema:
        return parse_openapi(state, llm)
    prompt = f"Provide help about OpenAPI based on user input: {state.user_input}"
    res = llm_call_helper(llm, prompt)
    state.text_content = TextContent(text=res)
    return update_scratchpad_reason(state, tool_name, "Provided OpenAPI help.")

def simulate_load_test(state: BotState, llm: Any) -> BotState:
    """Placeholder: simulate load test."""
    tool_name = BotIntent.SIMULATE_LOAD_TEST.value
    state.text_content = TextContent(text="Load test simulation not implemented.")
    return update_scratchpad_reason(state, tool_name, "Called simulate_load_test.")

def execute_workflow(state: BotState, llm: Any) -> BotState:
    """Placeholder: execute the API workflow."""
    tool_name = BotIntent.EXECUTE_WORKFLOW.value
    state.text_content = TextContent(text="Workflow execution not implemented.")
    return update_scratchpad_reason(state, tool_name, "Called execute_workflow.")

def unknown_intent(state: BotState, llm: Any) -> BotState:
    """Tool for unknown intents."""
    tool_name = BotIntent.UNKNOWN.value
    state.text_content = TextContent(text="Sorry, I didn't understand that. Please try again.")
    return update_scratchpad_reason(state, tool_name, "Unknown intent.")
