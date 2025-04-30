# tools.py
import json
from typing import Dict, Any, List
# Assume langchain.schema and langchain_openai are available
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI # Replace MockLLM with your actual LLM import
from models import BotState, GraphOutput, Node, Edge, TextContent, BotIntent, AVAILABLE_TOOLS
from pydantic import ValidationError # Import for handling potential JSON parsing errors

# --- Helper for programmatic cycle detection ---
# LLMs are not reliable for algorithmic tasks like cycle detection.
# This function performs a Depth First Search (DFS) to detect cycles.
def check_for_cycles(graph: GraphOutput) -> tuple[bool, str]:
    """Programmatically check for cycles in the graph using DFS."""
    nodes = {node.operationId for node in graph.nodes}
    # Build adjacency list representation
    adj: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    for edge in graph.edges:
        if edge.from_node in adj and edge.to_node in adj: # Only add edges between existing nodes
             adj[edge.from_node].append(edge.to_node)
        # Note: You might want error handling here if edges refer to non-existent nodes

    # 0: unvisited, 1: visiting (in current recursion stack), 2: visited (finished processing)
    visited = {}
    recursion_stack = {}

    def dfs(node_id):
        visited[node_id] = 1 # Mark as visiting
        recursion_stack[node_id] = True

        for neighbor_id in adj.get(node_id, []):
            if neighbor_id not in visited:
                if dfs(neighbor_id):
                    return True # Cycle detected in recursive call
            elif recursion_stack.get(neighbor_id):
                # Found a cycle: neighbor is in the current recursion stack
                return True

        recursion_stack[node_id] = False # Remove from recursion stack
        visited[node_id] = 2 # Mark as fully visited
        return False

    # Iterate through all nodes to cover disconnected components
    for node_id in nodes:
        if node_id not in visited:
            if dfs(node_id):
                return False, "Cycle detected in the graph." # Graph is invalid if cycle found

    return True, "No cycles detected." # Graph is valid


# --- Tool Functions ---
# Each tool function takes BotState and the LLM instance (which will be the worker LLM), and returns the updated BotState.
# The TOOL_FUNCTIONS dictionary is now defined within the LangGraphOpenApiRouter class.

def update_scratchpad_reason(state: BotState, tool_name: str, details: str) -> BotState:
     """Helper to append reasoning/details to the scratchpad."""
     current_reason = state.scratchpad.get('reason', '')
     # Limit the size of the scratchpad reason to avoid excessive context
     max_reason_length = 2000 # Adjust as needed
     new_entry = f"\nTool: {tool_name}\nDetails: {details}\n"
     combined_reason = current_reason + new_entry
     # Keep only the latest part if it exceeds max length
     state.scratchpad['reason'] = combined_reason[-max_reason_length:]
     return state

def parse_openapi(state: BotState, llm: Any) -> BotState:
    """Tool to parse OpenAPI spec text into a schema."""
    tool_name = BotIntent.OPENAPI_HELP.value # Using OPENAPI_HELP intent for this tool
    if not state.openapi_spec_text:
        state.text_content = TextContent(text="Error: No OpenAPI spec text provided to parse.")
        return update_scratchpad_reason(state, tool_name, "No spec text provided by user.")

    prompt = f"Parse this OpenAPI spec into a JSON schema object. Output only the JSON object.\nSpec:\n{state.openapi_spec_text}"

    try:
        # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
        res = llm.invoke(prompt) # Use the worker LLM passed to the tool

        schema = json.loads(res)
        state.openapi_schema = schema
        # Optional: Cache schema in scratchpad if needed for other steps
        # state.scratchpad['openapi_schema_cache'] = schema
        state.text_content = TextContent(text="OpenAPI schema parsed successfully.")
        return update_scratchpad_reason(state, tool_name, f"Parsed schema. Keys: {list(schema.keys()) if schema else 'None'}")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error parsing OpenAPI schema: {e}")
        return update_scratchpad_reason(state, tool_name, f"Parsing failed: {e}. LLM output: {res[:200]}...")


def generate_api_execution_graph(state: BotState, llm: Any) -> BotState:
    """Tool to generate an execution graph from the parsed schema."""
    tool_name = BotIntent.GENERATE_GRAPH.value
    if not state.openapi_schema:
        state.text_content = TextContent(text="Error: OpenAPI schema not available. Please parse the spec first.")
        return update_scratchpad_reason(state, tool_name, "Schema not available.")

    prompt = f"""Given this OpenAPI schema JSON, generate a likely API execution workflow as a Directed Acyclic Graph (DAG).
    Represent the graph as a JSON object matching the `GraphOutput` Pydantic model structure:
    {{
      "nodes": [{{ "operationId": str, "method": str, "path": str, "description": str, "payload": dict | null }}],
      "edges": [{{ "from_node": str, "to_node": str }}]
    }}
    Use the operationIds, methods, paths, and descriptions from the schema. Payload should be null initially for most nodes, maybe include a basic example for POST/PUT.
    Focus on common dependencies (e.g., login before other calls, create before get/update/delete).
    Ensure all operationIds used in edges exist as nodes. Output only the JSON object.

    OpenAPI Schema JSON:
    ```json
    {json.dumps(state.openapi_schema)}
    ```
    """
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Use the worker LLM passed to the tool

    try:
        graph_data = json.loads(res)
        # Validate and parse using Pydantic model
        graph_output = GraphOutput(**graph_data)
        state.graph_output = graph_output
        state.text_content = TextContent(text="Execution graph generated successfully.")
        return update_scratchpad_reason(state, tool_name, f"Generated graph with {len(graph_output.nodes)} nodes and {len(graph_output.edges)} edges.")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error generating execution graph: {e}. LLM output: {res[:200]}...")
        state.graph_output = None # Clear potentially invalid graph
        return update_scratchpad_reason(state, tool_name, f"Graph generation failed: {e}. LLM output: {res[:200]}...")


def generate_payload(state: BotState, llm: Any) -> BotState:
    """Tool to generate example payloads for operations in the graph and update nodes."""
    tool_name = BotIntent.GENERATE_PAYLOAD.value
    if not state.openapi_schema or not state.graph_output:
         state.text_content = TextContent(text="Error: Schema or graph not available to generate payloads.")
         return update_scratchpad_reason(state, tool_name, "Schema or graph missing.")

    prompt = f"""Given the OpenAPI schema and the current execution graph JSON, generate example JSON payloads for operations that require them (typically POST, PUT, PATCH).
    Update the `payload` field for the relevant nodes in the graph JSON and return the *entire* updated graph JSON object matching the `GraphOutput` structure. Operations without a request body (GET, DELETE, HEAD, OPTIONS) should keep payload as null.

    OpenAPI Schema JSON:
    ```json
    {json.dumps(state.openapi_schema)}
    ```

    Current Graph JSON:
    ```json
    {state.graph_output.model_dump_json()}
    ```
    Output only the updated graph JSON object.
    """
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Use the worker LLM passed to the tool

    try:
        graph_data = json.loads(res)
        updated_graph_output = GraphOutput(**graph_data) # Validate updated graph
        state.graph_output = updated_graph_output
        state.text_content = TextContent(text="Payloads generated and added to graph nodes.")
        # You might add logic here to count how many nodes got payloads
        return update_scratchpad_reason(state, tool_name, f"Generated payloads and updated graph.")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error generating payloads or updating graph: {e}. LLM output: {res[:200]}...")
        # Keep the old graph if update failed
        return update_scratchpad_reason(state, tool_name, f"Payload update failed: {e}. LLM output: {res[:200]}...")


def general_query(state: BotState, llm: Any) -> BotState:
    """Tool to answer general questions based on the schema or current state."""
    tool_name = BotIntent.GENERAL_QUERY.value
    if not state.openapi_schema:
        state.text_content = TextContent(text="Error: OpenAPI schema not available to answer questions.")
        return update_scratchpad_reason(state, tool_name, "Schema not available.")

    # Include user_input from state for the query
    prompt = f"""Answer the following question based on the provided OpenAPI schema JSON and the current bot state (including graph if available).
    If the schema/state doesn't contain the information, state that you cannot answer based on the available information.

    OpenAPI Schema JSON:
    ```json
    {json.dumps(state.openapi_schema)}
    ```
    Current Graph JSON (if available):
    ```json
    {state.graph_output.model_dump_json() if state.graph_output else 'null'}
    ```

    Question: {state.user_input}

    Output only the answer text.
    """
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Use the worker LLM passed to the tool

    state.text_content = TextContent(text=res)
    return update_scratchpad_reason(state, tool_name, f"Answered general query. Response length: {len(res)}")


def add_edge(state: BotState, llm: Any) -> BotState:
    """Tool to modify the execution graph by adding an edge based on user instruction."""
    tool_name = BotIntent.ADD_EDGE.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No execution graph available to add an edge.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")

    # Need to carefully prompt the LLM to parse the user instruction
    # and apply it to the GraphOutput JSON structure.
    # This is complex for an LLM - might require few-shot examples or a more structured input format.
    prompt = f"""You need to modify an existing API execution graph (DAG) based on a user's request.
    The graph is in JSON format matching the `GraphOutput` Pydantic model structure.
    The user wants to add a dependency (an edge) between two operationIds.
    Identify the 'from_node' and 'to_node' operationIds from the user's instruction and add the corresponding edge to the `edges` list.
    Ensure the resulting graph is still valid (no cycles) and that the operationIds exist as nodes.
    If the change creates a cycle, the operationIds are not in the graph, or the instruction is unclear, output an error message as JSON instead: {{ "error": "Reason why change failed" }}.
    Otherwise, output the *entire* updated graph JSON object matching the `GraphOutput` structure.

    Current Graph JSON:
    ```json
    {state.graph_output.model_dump_json()}
    ```

    User Instruction: "{state.user_input}"

    Output either the updated graph JSON or the error JSON.
    """
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Use the worker LLM passed to the tool

    try:
        response_data = json.loads(res)
        if "error" in response_data:
             state.text_content = TextContent(text=f"Error modifying graph: {response_data['error']}")
             return update_scratchpad_reason(state, tool_name, f"Modification failed: {response_data['error']}")
        else:
            # Validate and parse using Pydantic model
            updated_graph_output = GraphOutput(**response_data)
            # Optional: Perform programmatic cycle check again after LLM modification
            is_valid, validation_reason = check_for_cycles(updated_graph_output)
            if not is_valid:
                 state.text_content = TextContent(text=f"Error modifying graph: Change created a cycle. {validation_reason}")
                 return update_scratchpad_reason(state, tool_name, f"Modification failed: Created cycle. {validation_reason}")

            state.graph_output = updated_graph_output
            state.text_content = TextContent(text="Execution graph updated successfully.")
            return update_scratchpad_reason(state, tool_name, f"Added edge(s). New count: {len(updated_graph_output.edges)}")
    except (json.JSONDecodeError, ValidationError) as e:
        state.text_content = TextContent(text=f"Error parsing graph modification response: {e}. LLM output: {res[:200]}...")
        # Keep the old graph if parsing failed
        return update_scratchpad_reason(state, tool_name, f"Parsing modification response failed: {e}. LLM output: {res[:200]}...")


def validate_graph(state: BotState, llm: Any) -> BotState:
    """Tool to validate the current execution graph (e.g., check for cycles)."""
    tool_name = BotIntent.VALIDATE_GRAPH.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No execution graph available to validate.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")

    # Perform programmatic validation for cycles
    is_valid, validation_reason = check_for_cycles(state.graph_output)

    # Use LLM to provide a user-friendly explanation of the result
    prompt = f"""Based on the following graph validation result and the graph JSON, provide a user-friendly explanation of whether the graph is valid and why.

    Validation Result: {'Valid' if is_valid else 'Invalid'}
    Reason: {validation_reason}
    Graph JSON:
    ```json
    {state.graph_output.model_dump_json()}
    ```
    Output only the explanation text.
    """
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Use the worker LLM passed to the tool

    state.text_content = TextContent(text=f"Validation result: {'Valid' if is_valid else 'Invalid'}. {res}")
    return update_scratchpad_reason(state, tool_name, f"Graph validation performed. Valid: {is_valid}. Reason: {validation_reason[:100]}...")


def describe_execution_plan(state: BotState, llm: Any) -> BotState:
    """Tool to generate a natural language description of the current graph."""
    tool_name = BotIntent.DESCRIBE_GRAPH.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No execution graph available to describe.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")

    prompt = f"""Explain the workflow and dependencies represented by the following API execution graph JSON in a clear, user-friendly manner.
    Describe the sequence of operations and why certain steps depend on others based on typical API workflows.

    Graph JSON:
    ```json
    {state.graph_output.model_dump_json()}
    ```
    Output only the explanation text.
    """
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Use the worker LLM passed to the tool

    state.text_content = TextContent(text=res)
    return update_scratchpad_reason(state, tool_name, f"Generated graph description. Length: {len(res)}")


def get_execution_graph_json(state: BotState, llm: Any) -> BotState:
    """Tool to output the current execution graph JSON directly."""
    tool_name = BotIntent.GET_GRAPH_JSON.value
    if not state.graph_output:
        state.text_content = TextContent(text="Error: No execution graph available to output JSON.")
        return update_scratchpad_reason(state, tool_name, "Graph missing.")

    # Provide the JSON directly in text_content
    # Use model_dump_json for Pydantic object
    state.text_content = TextContent(text=state.graph_output.model_dump_json(indent=2))
    return update_scratchpad_reason(state, tool_name, "Outputted graph JSON.")


def openapi_help(state: BotState, llm: Any) -> BotState:
    """Tool to provide help about OpenAPI specs or handle initial spec parsing."""
    tool_name = BotIntent.OPENAPI_HELP.value
    # Check if the state contains spec text. If so, prioritize parsing.
    if state.openapi_spec_text and not state.openapi_schema:
         print("OPENAPI_HELP tool: Spec text found, attempting to parse.")
         # Call the parsing logic directly or delegate to a helper
         return parse_openapi(state, llm) # Use the worker LLM passed to this tool
    else:
        # Otherwise, provide general help about OpenAPI
        print("OPENAPI_HELP tool: Providing general help.")
        prompt = f"The user is asking for help about OpenAPI specifications based on their input: '{state.user_input}'. Provide helpful information."
        # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
        res = llm.invoke(prompt) # Use the worker LLM passed to the tool
        state.text_content = TextContent(text=res)
        return update_scratchpad_reason(state, tool_name, "Provided general OpenAPI help.")


# Placeholder tools - implement these if needed
def simulate_load_test(state: BotState, llm: Any) -> BotState:
     tool_name = BotIntent.SIMULATE_LOAD_TEST.value
     state.text_content = TextContent(text="Load test simulation is not yet implemented.")
     return update_scratchpad_reason(state, tool_name, "Called placeholder tool.")

def execute_workflow(state: BotState, llm: Any) -> BotState:
     tool_name = BotIntent.EXECUTE_WORKFLOW.value
     state.text_content = TextContent(text="Workflow execution is not yet implemented.")
     return update_scratchpad_reason(state, tool_name, "Called placeholder tool.")


def unknown_intent(state: BotState, llm: Any) -> BotState:
    """Tool for unknown intents."""
    tool_name = BotIntent.UNKNOWN.value
    state.text_content = TextContent(text="Sorry, I didn't understand that request. Please try rephrasing.")
    return update_scratchpad_reason(state, tool_name, "Unknown intent detected.")

# Note: TOOL_FUNCTIONS dictionary is now defined within LangGraphOpenApiRouter

