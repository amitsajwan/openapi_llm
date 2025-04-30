# tools.py
import json
from typing import Dict, Any
# Assume langchain.schema and langchain_openai are available
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI # Replace MockLLM with your actual LLM import
from models import BotState, GraphOutput, Node, Edge, TextContent, BotIntent, AVAILABLE_TOOLS
from pydantic import ValidationError # Import for handling potential JSON parsing errors

# --- Mock LLM Class for Demonstration ---
# IMPORTANT: Replace this MockLLM class with your actual LangChain LLM instance
# e.g., from langchain_openai import AzureChatOpenAI
# llm = AzureChatOpenAI(...)
class MockLLM:
    """
    A mock LLM for demonstration purposes.
    Replace this with your actual LangChain LLM instance.
    """
    def invoke(self, prompt: str) -> str:
        """Simulates an LLM call based on prompt keywords."""
        print(f"\n--- Mock LLM Call ---")
        print(f"Prompt: {prompt[:500]}...") # Print truncated prompt
        print("----------------------")
        # Simple rule-based mock responses based on prompt keywords
        if "Parse this OpenAPI spec" in prompt:
             # Mock schema parsing - simplified
            mock_schema = {
                "openapi": "3.0.0", "info": {"title": "Mock API"},
                "paths": {
                    "/users": {"get": {"operationId": "listUsers", "description": "Get users"}},
                    "/users/{id}": {"get": {"operationId": "getUser", "description": "Get single user"}},
                    "/login": {"post": {"operationId": "loginUser", "description": "Login"}},
                    "/items": {"get": {"operationId": "listItems", "description": "List items"}}
                }
            }
            print(f"Mock Response: {json.dumps(mock_schema)[:200]}...")
            return json.dumps(mock_schema)

        elif "generate an execution graph" in prompt:
            # Mock graph generation (GraphOutput format)
            mock_graph = {
                "nodes": [
                    {"operationId": "loginUser", "method": "POST", "path": "/login", "description": "Login"},
                    {"operationId": "listUsers", "method": "GET", "path": "/users", "description": "Get users"},
                    {"operationId": "getUser", "method": "GET", "path": "/users/{id}", "description": "Get single user"},
                    {"operationId": "listItems", "method": "GET", "path": "/items", "description": "List items"}
                ],
                "edges": [
                    {"from_node": "loginUser", "to_node": "listUsers"},
                    {"from_node": "loginUser", "to_node": "getUser"},
                    {"from_node": "loginUser", "to_node": "listItems"}
                ]
            }
            print(f"Mock Response: {json.dumps(mock_graph)[:200]}...")
            return json.dumps(mock_graph)

        elif "generate example JSON payloads" in prompt:
            # Mock payload generation and update graph
            # In a real LLM, this prompt is complex. Mocking simple update.
            try:
                # Extract current graph from prompt (simplified)
                graph_json_str = prompt.split("Current Graph JSON:")[1].split("Output only the updated graph JSON object.")[0].strip()
                graph_output_dict = json.loads(graph_json_str)
                graph_output = GraphOutput(**graph_output_dict)

                # Find the login node and add a mock payload
                for node in graph_output.nodes:
                    if node.operationId == "loginUser":
                        node.payload = {"username": "test_user", "password": "password123"}
                        break # Assume only one login node

                print(f"Mock Response (Payload Update): {graph_output.model_dump_json()[:200]}...")
                return graph_output.model_dump_json()
            except Exception as e:
                print(f"Mock Response (Payload Update Error): {{'error': '{str(e)}'}}")
                return "{}" # Return empty or error JSON

        elif "Modify this execution graph JSON" in prompt:
             # Mock graph modification - just add a dummy edge for demo
             # In real code, LLM would parse instruction and modify JSON intelligently
            try:
                graph_json_str = prompt.split("Current Graph JSON:")[1].split("User Instruction:")[0].strip()
                graph_output_dict = json.loads(graph_json_str)
                graph_output = GraphOutput(**graph_output_dict)

                # Simple mock: add an edge if user mentions 'getUser' and 'listItems'
                user_instruction = prompt.split("User Instruction:")[1].strip()
                if 'getUser' in user_instruction and 'listItems' in user_instruction:
                     new_edge = Edge(from_node="getUser", to_node="listItems")
                     if new_edge not in graph_output.edges: # Avoid duplicates in mock
                         graph_output.edges.append(new_edge)
                         print("Mock: Added edge getUser -> listItems")
                     else:
                          print("Mock: Edge already exists")
                else:
                     print("Mock: No specific edge instruction recognized")


                print(f"Mock Response (Modify Graph): {graph_output.model_dump_json()[:200]}...")
                return graph_output.model_dump_json()
            except Exception as e:
                print(f"Mock Response (Modify Error): {{'error': '{str(e)}'}}")
                return json.dumps({"error": f"Mock modification failed: {str(e)}"}) # Return error JSON format

        elif "Analyze the following execution graph JSON" in prompt:
            # Mock graph validation explanation
            # The actual cycle check is done programmatically in the tool
            is_valid = "No cycles detected" in prompt # Check based on programmatic result in prompt
            reason = "The graph structure looks good." if is_valid else "There seems to be a loop."
            print(f"Mock Response: '{reason}'")
            return reason # Mock LLM just explains the result

        elif "Explain the workflow represented by the following execution graph JSON" in prompt:
            # Mock explanation generation
            print(f"Mock Response: 'This is a mock explanation of the graph...'")
            return "This is a mock explanation of the graph based on the provided structure, highlighting dependencies like login before accessing user data."

        elif "Answer based on this OpenAPI schema" in prompt:
            # Mock general query answering
             print(f"Mock Response: 'This is a mock answer based on the schema...'")
             return "This is a mock answer based on the schema. It seems the API allows fetching user data and items after login."

        elif "Output the current execution graph JSON" in prompt:
             # Mock graph JSON output - extract from prompt
            graph_json_str = prompt.split("Graph JSON:")[1].strip()
            print(f"Mock Response: {graph_json_str[:200]}...")
            return graph_json_str

        elif "asking for help about OpenAPI specifications" in prompt:
             # Mock OpenAPI help
             print(f"Mock Response: 'OpenAPI specs describe RESTful APIs...'")
             return "OpenAPI specifications (formerly Swagger) are a standard, language-agnostic description format for RESTful APIs. They help define available endpoints, operations, parameters, and responses."

        elif "Available tools" in prompt:
             # This is the router prompt - mock intent selection
             # In real router, LLM logic decides based on user input & history
             user_input = prompt.split("User:")[1].split("Available tools:")[0].strip()
             if "create" in user_input.lower() or "generate" in user_input.lower() and "graph" in user_input.lower():
                 intent = BotIntent.GENERATE_GRAPH.value
             elif "add edge" in user_input.lower() or "modify graph" in user_input.lower():
                 intent = BotIntent.ADD_EDGE.value
             elif "explain" in user_input.lower() or "describe" in user_input.lower() and "graph" in user_input.lower():
                 intent = BotIntent.DESCRIBE_GRAPH.value
             elif "validate" in user_input.lower() and "graph" in user_input.lower():
                 intent = BotIntent.VALIDATE_GRAPH.value
             elif "payloads" in user_input.lower():
                  intent = BotIntent.GENERATE_PAYLOAD.value
             elif "show graph" in user_input.lower() or "get graph" in user_input.lower():
                  intent = BotIntent.GET_GRAPH_JSON.value
             elif "openapi help" in user_input.lower() or "spec help" in user_input.lower():
                  intent = BotIntent.OPENAPI_HELP.value
             else:
                 intent = BotIntent.GENERAL_QUERY.value # Default to general query

             print(f"Mock Response (Router Intent): {intent}")
             return intent # Router mock returns just the intent string

        # Default response for unhandled prompts
        print(f"Mock Response: 'Mock response for unexpected prompt.'")
        return "Mock response for unexpected prompt."

# Instantiate the mock LLM (REPLACE THIS WITH YOUR ACTUAL LLM)
llm = MockLLM()
# Example with AzureChatOpenAI (uncomment and configure if using)
# from langchain_openai import AzureChatOpenAI
# llm = AzureChatOpenAI(
#     azure_endpoint="YOUR_AZURE_ENDPOINT",
#     api_key="YOUR_API_KEY",
#     azure_deployment="YOUR_DEPLOYMENT_NAME",
#     api_version="YOUR_API_VERSION" # e.g., "2023-05-15"
# )

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
# Each tool function takes BotState and the LLM instance, and returns the updated BotState.

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
        res = llm.invoke(prompt) # Using mock LLM invoke

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
    res = llm.invoke(prompt) # Using mock LLM invoke

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
    res = llm.invoke(prompt) # Using mock LLM invoke

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
    res = llm.invoke(prompt) # Using mock LLM invoke

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
    res = llm.invoke(prompt) # Using mock LLM invoke

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
    explanation_text = llm.invoke(prompt) # Using mock LLM invoke

    state.text_content = TextContent(text=f"Validation result: {'Valid' if is_valid else 'Invalid'}. {explanation_text}")
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
    res = llm.invoke(prompt) # Using mock LLM invoke

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
    """Tool to provide help about OpenAPI specs."""
    tool_name = BotIntent.OPENAPI_HELP.value
    prompt = f"The user is asking for help about OpenAPI specifications based on their input: '{state.user_input}'. Provide helpful information."
    # res = llm([HumanMessage(content=prompt)]).content # Use this with actual LangChain
    res = llm.invoke(prompt) # Using mock LLM invoke
    state.text_content = TextContent(text=res)
    return update_scratchpad_reason(state, tool_name, "Provided OpenAPI help.")


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


# Map BotIntent enum values (string values) to tool functions
TOOL_FUNCTIONS = {
    BotIntent.GENERATE_GRAPH.value: generate_api_execution_graph,
    BotIntent.GENERAL_QUERY.value: general_query,
    BotIntent.OPENAPI_HELP.value: openapi_help, # Re-purposed for initial parsing as well
    BotIntent.GENERATE_PAYLOAD.value: generate_payload,
    BotIntent.ADD_EDGE.value: add_edge,
    BotIntent.VALIDATE_GRAPH.value: validate_graph,
    BotIntent.DESCRIBE_GRAPH.value: describe_execution_plan,
    BotIntent.GET_GRAPH_JSON.value: get_execution_graph_json,
    BotIntent.SIMULATE_LOAD_TEST.value: simulate_load_test,
    BotIntent.EXECUTE_WORKFLOW.value: execute_workflow,
    BotIntent.UNKNOWN.value: unknown_intent,
}

# Assert that all defined AVAILABLE_TOOLS have a corresponding function mapping
assert set(AVAILABLE_TOOLS).issubset(TOOL_FUNCTIONS.keys()), "Not all AVAILABLE_TOOLS have a function mapping!"
# Assert that the UNKNOWN intent also has a mapping
assert BotIntent.UNKNOWN.value in TOOL_FUNCTIONS, "UNKNOWN intent is not mapped to a function!"

