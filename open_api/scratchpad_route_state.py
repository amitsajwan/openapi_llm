# models.py
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Literal, Type, TypeVar, Union
from enum import Enum

# Define available intents (tools/actions) explicitly
class BotIntent(str, Enum):
    GENERATE_GRAPH = "generate_api_execution_graph"
    GENERAL_QUERY = "general_query"
    OPENAPI_HELP = "openapi_help" # Handles initial parsing and general help
    GENERATE_PAYLOAD = "generate_payload"
    ADD_EDGE = "add_edge"
    VALIDATE_GRAPH = "validate_graph"
    DESCRIBE_GRAPH = "describe_execution_plan"
    GET_GRAPH_JSON = "get_execution_graph_json"
    SIMULATE_LOAD_TEST = "simulate_load_test" # Placeholder intent
    EXECUTE_WORKFLOW = "execute_workflow"   # Placeholder intent
    UNKNOWN = "unknown"
    # Internal intents (not directly routed from user input, but used in graph)
    EXTRACT_PARAMS = "extract_parameters" # New: for parameter extraction step

class Node(BaseModel):
    """Represents a single operation/node in the execution graph."""
    operationId: str
    # Using str for method, as LLMs sometimes struggle with strict Literal output
    method: str #Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    path: str
    description: str
    payload: Optional[Dict[str, Any]] = Field(default=None) # Use Dict for JSON payload example

class Edge(BaseModel):
    """Represents a dependency (edge) between two nodes."""
    from_node: str # operationId of the source node
    to_node: str   # operationId of the target node

class GraphOutput(BaseModel):
    """Represents the structured execution graph."""
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

class TextContent(BaseModel):
    """Represents user-facing text output from the bot."""
    text: str = ""

# --- Tool Parameter Models ---
# Define Pydantic models for expected parameters for tools that need them.
# This makes parameter extraction more structured.

class AddEdgeParams(BaseModel):
    """Parameters for the add_edge tool."""
    from_node: str = Field(description="The operationId of the node the edge starts from.")
    to_node: str = Field(description="The operationId of the node the edge goes to.")

class GeneratePayloadParams(BaseModel):
    """Parameters for the generate_payload tool (if targeting specific nodes)."""
    # Example: Could allow specifying which operationIds to generate payloads for
    operation_ids: Optional[List[str]] = Field(default=None, description="Optional list of operationIds to generate payloads for. If null, generate for all relevant operations.")

# Add other parameter models as needed for other tools

# --- Main Bot State ---
class BotState(BaseModel):
    """Represents the state of the bot during a conversation turn."""
    # Unique identifier for the conversation session (for caching)
    session_id: str

    # User input for the current turn
    user_input: Optional[str] = None

    # Determined intent for the current turn (after routing)
    intent: BotIntent = BotIntent.UNKNOWN

    # Raw spec text (initial input)
    openapi_spec_text: Optional[str] = None
    # Parsed schema cached
    # Use Dict[str, Any] as OpenAPI schema structure can be complex and varies
    openapi_schema: Optional[Dict[str, Any]] = Field(default=None)

    # The generated execution graph using the structured format
    graph_output: Optional[GraphOutput] = Field(default=None)

    # Generated payloads (if not stored within nodes) - kept separate for flexibility
    # This might become less necessary if payloads are stored in Node.payload
    payloads_output: Optional[Dict[str, Any]] = Field(default=None)

    # User-facing text content explaining graph, answer to query, etc.
    text_content: TextContent = Field(default_factory=TextContent)

    # History of (user_input, intent) - useful for debugging/tracing
    action_history: List[tuple] = Field(default_factory=list)

    # Scratchpad for intermediate LLM reasoning, cached results, detailed logs
    # This is crucial for the router's context over turns
    scratchpad: Dict[str, Any] = Field(default_factory=dict)

    # --- New Fields for Parameter Extraction ---
    # The Pydantic model type expected for parameter extraction for the current intent
    expected_params_model: Optional[str] = Field(default=None, description="Name of the Pydantic model expected for parameter extraction (e.g., 'AddEdgeParams').")
    # The extracted parameters, stored as a dictionary after extraction
    extracted_params: Optional[Dict[str, Any]] = Field(default=None, description="Extracted parameters for the current intent.")
    # --- End New Fields ---

    # You might add other state variables here, e.g.,
    # error: Optional[str] = None
    # current_plan: List[BotIntent] = Field(default_factory=list) # If implementing complex planning
    # plan_step_index: int = 0

# Helper to get the list of available tool names (excluding UNKNOWN and internal intents)
# This list is used to tell the router LLM which tools it can choose from.
AVAILABLE_TOOLS = [e.value for e in BotIntent if e not in [BotIntent.UNKNOWN, BotIntent.SIMULATE_LOAD_TEST, BotIntent.EXECUTE_WORKFLOW, BotIntent.EXTRACT_PARAMS]]

# Mapping from BotIntent value to the expected parameter model type
# Used by the parameter extraction node
INTENT_PARAMETER_MODELS: Dict[str, Type[BaseModel]] = {
    BotIntent.ADD_EDGE.value: AddEdgeParams,
    # Add other intents requiring specific parameters here
    # BotIntent.GENERATE_PAYLOAD.value: GeneratePayloadParams,
}

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

def load_state(session_id: str, cache_dir: str) -> Optional[BotState]:
    """Loads BotState from a JSON file in the cache directory."""
    file_path = os.path.join(cache_dir, f"{session_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            # Manually handle enum conversion if needed, or ensure Pydantic handles it
            # Pydantic should handle str -> Enum for fields typed as Enum
            return BotState(**state_data)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error loading state for session {session_id}: {e}")
            # Optionally delete corrupted file
            # os.remove(file_path)
            return None
    return None

def save_state(state: BotState, cache_dir: str):
    """Saves BotState to a JSON file in the cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"{state.session_id}.json")
    try:
        # Use model_dump_json for Pydantic model serialization
        with open(file_path, 'w') as f:
            f.write(state.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error saving state for session {state.session_id}: {e}")

