# models.py
# No changes needed in the models file for this update.
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

# Define available intents (tools/actions) explicitly
class BotIntent(str, Enum):
    GENERATE_GRAPH = "generate_api_execution_graph"
    GENERAL_QUERY = "general_query"
    OPENAPI_HELP = "openapi_help"
    GENERATE_PAYLOAD = "generate_payload"
    ADD_EDGE = "add_edge"
    VALIDATE_GRAPH = "validate_graph"
    DESCRIBE_GRAPH = "describe_execution_plan"
    GET_GRAPH_JSON = "get_execution_graph_json"
    SIMULATE_LOAD_TEST = "simulate_load_test" # Placeholder intent
    EXECUTE_WORKFLOW = "execute_workflow"   # Placeholder intent
    UNKNOWN = "unknown"

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

# This will be our primary state object for LangGraph
class BotState(BaseModel):
    """Represents the state of the bot during a conversation turn."""
    # User input for the current turn
    user_input: Optional[str] = None

    # Determined intent for the current turn
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

    # You might add other state variables here, e.g.,
    # error: Optional[str] = None
    # current_plan: List[BotIntent] = Field(default_factory=list) # If implementing complex planning
    # plan_step_index: int = 0

# Helper to get the list of available tool names (excluding UNKNOWN and placeholders)
# This list is used to tell the router LLM which tools it can choose from.
AVAILABLE_TOOLS = [e.value for e in BotIntent if e not in [BotIntent.UNKNOWN, BotIntent.SIMULATE_LOAD_TEST, BotIntent.EXECUTE_WORKFLOW]]

