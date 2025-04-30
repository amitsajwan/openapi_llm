# openapi_agent.py
import os
import json
import logging
from typing import Dict, Any, List, Optional, Type
# Assume langchain.schema and langchain_openai are available
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from models import ( # Import necessary models and helpers
    BotState, BotIntent, AVAILABLE_TOOLS, GraphOutput, Node, Edge, TextContent,
    INTENT_PARAMETER_MODELS, AddEdgeParams, load_state, save_state # Import new models/helpers
)
from tools import ( # Import tool functions and helpers (excluding TOOL_FUNCTIONS dict)
    update_scratchpad_reason,
    # parse_openapi, # parse_openapi is called internally by openapi_help
    generate_api_execution_graph,
    generate_payload,
    general_query,
    add_edge,
    validate_graph,
    describe_execution_plan,
    get_execution_graph_json,
    openapi_help, # This tool now acts as the entry for parsing and help
    simulate_load_test,
    execute_workflow,
    unknown_intent,
    check_for_cycles # Import the helper function
)
from pydantic import BaseModel, ValidationError # Import BaseModel and ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangGraphOpenApiRouter:
    """
    Encapsulates the LangGraph-based OpenAPI bot logic.
    Handles initialization, tool registration, graph building, and invocation.
    Uses separate router and worker LLMs and includes state caching and parameter extraction.
    """
    def __init__(self, session_id: str, spec_text: Optional[str], llm_router: Any, llm_worker: Any, cache_dir: str = "./agent_cache"):
        """
        Initializes the OpenAPI agent.

        Args:
            session_id: A unique identifier for the conversation session (for caching).
            spec_text: The raw OpenAPI specification text (can be None if loading from cache).
            llm_router: The LLM instance used for routing decisions.
            llm_worker: The LLM instance used by the tool functions.
            cache_dir: Directory for caching session states.
        """
        self.session_id = session_id
        self.spec_text = spec_text # Initial spec text, might be None if loaded from cache
        self.llm_router = llm_router
        self.llm_worker = llm_worker # Store the worker LLM
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Agent initialized for session: {self.session_id}, cache dir: {self.cache_dir}")

        # Define the mapping of BotIntent values (tool names) to tool functions
        # These functions will be called by the graph nodes
        self.tools: Dict[str, Any] = {
            BotIntent.GENERATE_GRAPH.value: generate_api_execution_graph,
            BotIntent.GENERAL_QUERY.value: general_query,
            BotIntent.OPENAPI_HELP.value: openapi_help, # This tool handles both parsing and general help
            BotIntent.GENERATE_PAYLOAD.value: generate_payload,
            BotIntent.ADD_EDGE.value: add_edge,
            BotIntent.VALIDATE_GRAPH.value: validate_graph,
            BotIntent.DESCRIBE_GRAPH.value: describe_execution_plan,
            BotIntent.GET_GRAPH_JSON.value: get_execution_graph_json,
            BotIntent.SIMULATE_LOAD_TEST.value: simulate_load_test,
            BotIntent.EXECUTE_WORKFLOW.value: execute_workflow,
            BotIntent.UNKNOWN.value: unknown_intent,
            # Internal node for parameter extraction
            BotIntent.EXTRACT_PARAMS.value: self._extract_parameters_step,
        }

        # Assert that all defined AVAILABLE_TOOLS have a corresponding function mapping
        assert set(AVAILABLE_TOOLS).issubset(self.tools.keys()), "Not all AVAILABLE_TOOLS have a function mapping!"
        # Assert that the UNKNOWN and EXTRACT_PARAMS intents also have a mapping
        assert BotIntent.UNKNOWN.value in self.tools, "UNKNOWN intent is not mapped to a function!"
        assert BotIntent.EXTRACT_PARAMS.value in self.tools, "EXTRACT_PARAMS intent is not mapped to a function!"


        logger.info("Registered Tools: %s", list(self.tools.keys()))

        # Build the LangGraph application
        self.graph_app = self._build_graph()
        logger.info("LangGraph application built.")

    def _route_step(self, state: BotState) -> str:
        """
        LangGraph node: Determines the next tool/intent using the router LLM.
        Returns the string name (value) of the next BotIntent or internal node ('extract_parameters').
        """
        print("\n--- Routing Step ---")
        # Construct the prompt for the router LLM
        # Include scratchpad history for context
        history_prompt = state.scratchpad.get("reason", "")
        if history_prompt:
            # Add a header for clarity in the prompt
            history_prompt = "--- Conversation History and Reasoning ---\n" + history_prompt + "\n------------------------------------------\n"

        # Include available tools explicitly in the prompt
        # Exclude internal tools like EXTRACT_PARAMS from the list presented to the router LLM
        routable_tools = [t for t in AVAILABLE_TOOLS if t != BotIntent.EXTRACT_PARAMS.value]
        tool_list_str = ", ".join(routable_tools)

        prompt = f"""{history_prompt}
User: {state.user_input}

Review the conversation history and the user's latest request.
Decide the single most appropriate action (tool) the bot should take next from the list below.
Respond with *only* the exact string name of the chosen tool.
If the user's intent is unclear or doesn't map to an available tool, respond with '{BotIntent.UNKNOWN.value}'.

Available tools: {tool_list_str}

Output:""" # Expecting just the tool name string


        # Add the current user input to the reasoning *before* calling the LLM
        # This logs the input even if routing fails
        user_input_reason = f"User Input: {state.user_input}\n"
        state.scratchpad['reason'] = state.scratchpad.get('reason', '') + user_input_reason
        print(f"Routing Prompt:\n{prompt[:500]}...") # Print truncated prompt

        # Call the router LLM
        # In a real scenario: res = self.llm_router([HumanMessage(content=prompt)]).content
        res = self.llm_router.invoke(prompt).strip() # Use invoke and strip whitespace

        # Determine the next step based on the LLM's output
        # Check if the LLM's response is a valid tool name defined in self.tools
        # Note: The router LLM should only output names from AVAILABLE_TOOLS (excluding internal ones)
        # We check against self.tools.keys() which includes internal ones for safety,
        # but the prompt guides it to only choose from routable_tools.
        next_step_name = res if res in self.tools else BotIntent.UNKNOWN.value

        # Update state with the chosen intent
        try:
            # Validate the chosen name against the BotIntent Enum
            state.intent = BotIntent(next_step_name)
        except ValueError:
            # Fallback if LLM output is not a valid enum value
            state.intent = BotIntent.UNKNOWN
            next_step_name = BotIntent.UNKNOWN.value # Ensure the return value matches

        # Add the router's decision to the reasoning scratchpad
        state.scratchpad['reason'] += f"Router Decision: Chose tool '{state.intent.value}'.\n"

        # Record action history for tracing
        state.action_history.append((state.user_input, state.intent.value))

        print(f"Router decided: {state.intent.value}")

        # --- Parameter Extraction Logic ---
        # After routing, check if the chosen intent requires parameter extraction.
        if state.intent.value in INTENT_PARAMETER_MODELS:
             logger.info(f"Intent '{state.intent.value}' requires parameter extraction.")
             # Store the name of the expected parameter model in state
             state.expected_params_model = INTENT_PARAMETER_MODELS[state.intent.value].__name__
             # Route to the parameter extraction node
             return BotIntent.EXTRACT_PARAMS.value
        else:
             # No parameter extraction needed, route directly to the tool node
             state.expected_params_model = None # Clear any previous expectation
             state.extracted_params = None # Clear any previous extracted params
             return state.intent.value # Return the chosen tool name

    def _extract_parameters_step(self, state: BotState) -> str:
        """
        LangGraph node: Extracts parameters from user input using the worker LLM
        based on the expected parameter model for the current intent.
        Returns the name of the actual tool node to execute next (state.intent.value).
        """
        print("\n--- Parameter Extraction Step ---")
        tool_name = BotIntent.EXTRACT_PARAMS.value # Name of this node/tool

        if not state.user_input:
             state.text_content = TextContent(text="Error: No user input available for parameter extraction.")
             return update_scratchpad_reason(state, tool_name, "No user input.").intent.value # Go to UNKNOWN or handle error

        expected_model_name = state.expected_params_model
        if not expected_model_name:
             # This should not happen if routing logic is correct, but as a safeguard
             state.text_content = TextContent(text="Internal Error: Parameter extraction node reached without an expected model.")
             # Route to the original intended tool anyway, it should handle missing params
             return update_scratchpad_reason(state, tool_name, "No expected model specified in state.").intent.value

        # Get the actual Pydantic model type from the name
        # This requires access to the models defined in models.py
        # A safer way might be to pass the model type itself in the state,
        # or have a lookup dictionary here. Let's use a lookup dictionary.
        param_model_type = INTENT_PARAMETER_MODELS.get(state.intent.value)

        if not param_model_type:
             # Should not happen based on routing, but safeguard
             state.text_content = TextContent(text=f"Internal Error: No parameter model found for intent '{state.intent.value}'.")
             return update_scratchpad_reason(state, tool_name, f"No model found for intent {state.intent.value}.").intent.value


        # Construct prompt for the worker LLM to extract parameters
        prompt = f"""Extract parameters from the user's request based on the following JSON schema.
        The user's request is: "{state.user_input}"
        The parameters should match the schema for the '{state.intent.value}' tool.
        Output the extracted parameters as a JSON object matching the schema.
        If a parameter cannot be found in the user's request, omit it or set it to null if the schema allows.
        Output only the JSON object.

        Parameter Schema (Pydantic model '{expected_model_name}'):
        ```json
        {json.dumps(param_model_type.model_json_schema())}
        ```
        """

        # Call the worker LLM for extraction
        # res = self.llm_worker([HumanMessage(content=prompt)]).content # Use this with actual LangChain
        res = self.llm_worker.invoke(prompt).strip() # Use invoke and strip whitespace

        try:
            extracted_data = json.loads(res)
            # Validate the extracted data against the Pydantic model
            validated_params = param_model_type(**extracted_data)
            state.extracted_params = validated_params.model_dump() # Store validated params as dict

            logger.info(f"Successfully extracted parameters for '{state.intent.value}': {state.extracted_params}")
            update_scratchpad_reason(state, tool_name, f"Extracted params for '{state.intent.value}': {state.extracted_params}")

            # Now route to the actual tool node
            return state.intent.value

        except (json.JSONDecodeError, ValidationError) as e:
            state.text_content = TextContent(text=f"Error extracting parameters for '{state.intent.value}': {e}. Please rephrase your request.")
            state.extracted_params = None # Clear potentially invalid params
            update_scratchpad_reason(state, tool_name, f"Parameter extraction failed for '{state.intent.value}': {e}. LLM output: {res[:200]}...")
            # If extraction fails, route to UNKNOWN or back to route?
            # Routing to UNKNOWN seems reasonable if the required info wasn't provided.
            state.intent = BotIntent.UNKNOWN # Update intent to unknown
            return BotIntent.UNKNOWN.value # Route to the unknown tool

        except Exception as e:
             state.text_content = TextContent(text=f"An unexpected error occurred during parameter extraction: {e}")
             state.extracted_params = None
             update_scratchpad_reason(state, tool_name, f"Unexpected error during extraction: {e}")
             state.intent = BotIntent.UNKNOWN # Update intent to unknown
             return BotIntent.UNKNOWN.value # Route to the unknown tool


    def _build_graph(self):
        """
        Builds the LangGraph StateGraph.
        Uses the stored llm_worker instance for tool nodes.
        Includes the parameter extraction node.
        """
        # The graph state is our Pydantic BotState model
        builder = StateGraph(BotState)

        # Add the router node
        builder.add_node("route", self._route_step) # This node uses self.llm_router

        # Add the parameter extraction node
        builder.add_node(BotIntent.EXTRACT_PARAMS.value, self._extract_parameters_step) # This node uses self.llm_worker internally

        # Add nodes for each tool function
        # Each tool function receives and returns the BotState
        # We wrap the tool function in a lambda to pass the worker LLM instance (self.llm_worker)
        # The EXTRACT_PARAMS tool is handled separately above.
        tool_nodes = {
            name: lambda state, f=func: f(state, self.llm_worker)
            for name, func in self.tools.items()
            if name != BotIntent.EXTRACT_PARAMS.value # Exclude the extraction node here
        }

        for name, node in tool_nodes.items():
             builder.add_node(name, node)

        # Define the entry point of the graph
        builder.add_edge(START, "route")

        # Define the transitions from the router node
        # If the router decides on an intent that needs parameters, go to EXTRACT_PARAMS.
        # Otherwise, go directly to the tool node.
        def route_decision(state: BotState) -> str:
             # The _route_step already decided state.intent and returned the next node name
             # If _route_step returned EXTRACT_PARAMS, we go there.
             # Otherwise, we go to the tool name returned by _route_step.
             return state.intent.value if state.intent.value in INTENT_PARAMETER_MODELS else state.intent.value # This logic is now handled within _route_step return

        # The conditional edge from 'route' uses the string name returned by _route_step
        builder.add_conditional_edges(
            "route",
            lambda state: state.intent.value, # Use the intent value determined in _route_step
            # Map the intent value to the corresponding node name
            self.tools # Directly use the self.tools dict as the mapping
        )

        # Define transitions from the parameter extraction node
        # After extraction, route to the original intended tool node (state.intent.value)
        # If extraction failed and intent was updated to UNKNOWN, it will route to UNKNOWN.
        builder.add_edge(BotIntent.EXTRACT_PARAMS.value, "route") # After extraction, go back to route to let it transition to the correct tool

        # Define transitions from the tool nodes
        # Most tools should transition back to the router for the next turn,
        # allowing the bot to process follow-up questions or actions.
        # Some tools (like unknown or get_graph_json) might end the conversation turn.
        for tool_name in self.tools.keys():
             if tool_name in [BotIntent.UNKNOWN.value, BotIntent.EXTRACT_PARAMS.value]:
                  # These nodes don't transition back to themselves or other tools directly
                  pass # Handled by specific edges
             elif tool_name == BotIntent.GET_GRAPH_JSON.value:
                  # Outputting JSON is likely the final action for that specific request.
                  builder.add_edge(tool_name, END)
             elif tool_name == BotIntent.GENERAL_QUERY.value:
                 # A general query might be a one-off question, so end the turn.
                 builder.add_edge(tool_name, END)
             # Note: openapi_help now goes back to route to allow follow-up after parsing/help
             # elif tool_name == BotIntent.OPENAPI_HELP.value:
             #     builder.add_edge(tool_name, END) # Example: end after help

             else:
                  # Most other tools (generate_graph, add_edge, describe_graph, generate_payload, openapi_help)
                  # likely require further interaction or explanation, so go back to the router.
                  builder.add_edge(tool_name, "route")

        # Compile the graph
        app = builder.compile()
        return app

    def invoke_graph(self, user_input: str, session_id: Optional[str] = None) -> BotState:
        """
        Invokes the LangGraph application with new user input.
        Loads state from cache if session_id is provided, saves state after execution.

        Args:
            user_input: The user's input for the current turn.
            session_id: Optional unique identifier for the conversation session.
                        If None, a new session starts without loading previous state.

        Returns:
            The updated BotState after running the graph.
        """
        current_state: Optional[BotState] = None
        if session_id:
             # Attempt to load state for the given session ID
             current_state = load_state(session_id, self.cache_dir)
             if current_state:
                  logger.info(f"Loaded state for session: {session_id}")
             else:
                  logger.info(f"No saved state found for session: {session_id}. Starting new session.")
                  # If no state found, create a new one with the provided session_id
                  current_state = BotState(session_id=session_id, openapi_spec_text=self.spec_text)
        else:
             # If no session_id provided, start a new transient session
             logger.info("No session_id provided. Starting a new transient session.")
             current_state = BotState(session_id=os.urandom(16).hex(), openapi_spec_text=self.spec_text) # Generate a random ID

        # Update the state with the new user input for the current turn
        initial_state = current_state.copy() # Create a mutable copy
        initial_state.user_input = user_input
        # Clear previous text content so the new tool generates fresh output
        initial_state.text_content = TextContent(text="")
        # Reset intent and extracted params as they will be determined by the router/extraction step
        initial_state.intent = BotIntent.UNKNOWN
        initial_state.expected_params_model = None
        initial_state.extracted_params = None


        logger.info(f"Invoking graph with user input: '{user_input}' for session: {initial_state.session_id}")
        # Invoke the compiled graph with the initial state for this turn
        # The graph will run until it reaches an END node
        try:
            final_state = self.graph_app.invoke(initial_state)
            logger.info(f"Graph invocation finished. Final intent: {final_state.intent.value}")

            # Save the final state if a session_id was used
            if session_id: # Only save if the user provided a session_id
                 save_state(final_state, self.cache_dir)
                 logger.info(f"State saved for session: {final_state.session_id}")

            return final_state
        except Exception as e:
            logger.error(f"Error during graph invocation for session {initial_state.session_id}: {e}", exc_info=True)
            # Update state with error message before returning
            error_state = initial_state.copy() # Start from the state before the error occurred
            error_state.text_content = TextContent(text=f"An error occurred: {e}")
            # Ensure intent is marked as unknown after an error
            error_state.intent = BotIntent.UNKNOWN
            # Save the error state if session_id was used
            if session_id:
                 save_state(error_state, self.cache_dir)
                 logger.info(f"Error state saved for session: {error_state.session_id}")

            return error_state

