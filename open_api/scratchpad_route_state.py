# openapi_agent.py
import os
import json
import logging
from typing import Dict, Any, List, Optional
# Assume langchain.schema and langchain_openai are available
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from models import BotState, BotIntent, AVAILABLE_TOOLS, GraphOutput, Node, Edge, TextContent # Import necessary models
from tools import ( # Import tool functions and helpers (excluding TOOL_FUNCTIONS dict)
    update_scratchpad_reason,
    parse_openapi,
    generate_api_execution_graph,
    generate_payload,
    general_query,
    add_edge,
    validate_graph,
    describe_execution_plan,
    get_execution_graph_json,
    openapi_help,
    simulate_load_test,
    execute_workflow,
    unknown_intent,
    check_for_cycles # Import the helper function
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangGraphOpenApiRouter:
    """
    Encapsulates the LangGraph-based OpenAPI bot logic.
    Handles initialization, tool registration, graph building, and invocation.
    """
    def __init__(self, spec_text: str, llm_router: Any, llm_worker: Any, cache_dir: str = "./cache"):
        """
        Initializes the OpenAPI agent.

        Args:
            spec_text: The raw OpenAPI specification text.
            llm_router: The LLM instance used for routing decisions.
            llm_worker: The LLM instance used by the tool functions.
            cache_dir: Directory for caching (currently unused but kept for structure).
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.spec_text = spec_text
        self.llm_router = llm_router
        self.llm_worker = llm_worker # Store the worker LLM
        self.cache_dir = cache_dir

        # Define the mapping of BotIntent values (tool names) to tool functions
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
        }

        # Assert that all defined AVAILABLE_TOOLS have a corresponding function mapping
        assert set(AVAILABLE_TOOLS).issubset(self.tools.keys()), "Not all AVAILABLE_TOOLS have a function mapping!"
        # Assert that the UNKNOWN intent also has a mapping
        assert BotIntent.UNKNOWN.value in self.tools, "UNKNOWN intent is not mapped to a function!"

        logger.info("Registered Tools: %s", list(self.tools.keys()))

        # Build the LangGraph application
        self.graph_app = self._build_graph()
        logger.info("LangGraph application built.")

    def _route_step(self, state: BotState) -> str:
        """
        LangGraph node: Determines the next tool/intent using the router LLM.
        Returns the string name (value) of the next BotIntent.
        """
        print("\n--- Routing Step ---")
        # Construct the prompt for the router LLM
        # Include scratchpad history for context
        history_prompt = state.scratchpad.get("reason", "")
        if history_prompt:
            # Add a header for clarity in the prompt
            history_prompt = "--- Conversation History and Reasoning ---\n" + history_prompt + "\n------------------------------------------\n"

        # Include available tools explicitly in the prompt
        tool_list_str = ", ".join(AVAILABLE_TOOLS) # Use the list from models

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
        next_step_name = res if res in self.tools else BotIntent.UNKNOWN.value

        # Update state with the chosen intent
        try:
            # Validate the chosen name against the BotIntent Enum
            state.intent = BotIntent(next_step_name)
        except ValueError:
            # Fallback if LLM output is not a valid enum value (should be caught by `res in self.tools` but double-checking)
            state.intent = BotIntent.UNKNOWN
            next_step_name = BotIntent.UNKNOWN.value # Ensure the return value matches

        # Add the router's decision to the reasoning scratchpad
        state.scratchpad['reason'] += f"Router Decision: Chose tool '{state.intent.value}'.\n"

        # Record action history for tracing
        state.action_history.append((state.user_input, state.intent.value))

        print(f"Router decided: {state.intent.value}")
        # The router node returns the string name of the *next node* (the chosen tool)
        # LangGraph will use this return value to follow the conditional edge.
        return state.intent.value

    def _build_graph(self):
        """
        Builds the LangGraph StateGraph.
        Uses the stored llm_worker instance for tool nodes.
        """
        # The graph state is our Pydantic BotState model
        builder = StateGraph(BotState)

        # Add the router node
        builder.add_node("route", self._route_step) # This node uses self.llm_router

        # Add nodes for each tool function
        # Each tool function receives and returns the BotState
        # We wrap the tool function in a lambda to pass the worker LLM instance (self.llm_worker)
        tool_nodes = {name: lambda state, f=func: f(state, self.llm_worker) for name, func in self.tools.items()}

        for name, node in tool_nodes.items():
             builder.add_node(name, node)

        # Define the entry point of the graph
        builder.add_edge(START, "route")

        # Define the transitions from the router node
        # The conditional edge uses the return value of the 'route_step' node
        # which is the string name of the chosen tool.
        # The mapping {name: name for name in self.tools.keys()}
        # means if route_step returns "tool_name", the graph transitions to the node named "tool_name".
        builder.add_conditional_edges(
            "route",
            lambda next_step_name: next_step_name, # The next state key is the return value of route_step
            # Map the string name returned by route_step to the corresponding node name
            # Ensure this mapping covers all possible string outputs of route_step
            self.tools # Directly use the self.tools dict as the mapping
        )

        # Define transitions from the tool nodes
        # Most tools should transition back to the router for the next turn,
        # allowing the bot to process follow-up questions or actions.
        # Some tools (like unknown or get_graph_json) might end the conversation turn.
        for tool_name in self.tools.keys():
             if tool_name == BotIntent.UNKNOWN.value:
                  # If intent is unknown, end the current turn.
                  builder.add_edge(tool_name, END)
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

    def invoke_graph(self, user_input: str, current_state: Optional[BotState] = None) -> BotState:
        """
        Invokes the LangGraph application with new user input and the current state.

        Args:
            user_input: The user's input for the current turn.
            current_state: The BotState from the previous turn, or None for the first turn.

        Returns:
            The updated BotState after running the graph.
        """
        # If no current state is provided, create a new initial state
        if current_state is None:
            logger.info("Creating new initial state.")
            initial_state = BotState(
                user_input=user_input,
                # Pass the initial spec text here if it's the very first turn
                # In a real app, you might handle spec upload separately or check state
                openapi_spec_text=self.spec_text # Assuming spec_text is provided at init
            )
        else:
            # Update the existing state with the new user input
            logger.info("Updating existing state with new user input.")
            initial_state = current_state.copy() # Create a mutable copy
            initial_state.user_input = user_input
            # Clear previous text content so the new tool generates fresh output
            initial_state.text_content = TextContent(text="")
            # Reset intent as it will be determined by the router
            initial_state.intent = BotIntent.UNKNOWN


        logger.info(f"Invoking graph with user input: '{user_input}'")
        # Invoke the compiled graph with the initial state for this turn
        # The graph will run until it reaches an END node
        try:
            final_state = self.graph_app.invoke(initial_state)
            logger.info(f"Graph invocation finished. Final intent: {final_state.intent.value}")
            return final_state
        except Exception as e:
            logger.error(f"Error during graph invocation: {e}", exc_info=True)
            # Optionally update state with error message before returning
            if current_state:
                error_state = current_state.copy()
                error_state.text_content = TextContent(text=f"An error occurred: {e}")
                return error_state
            else:
                 # If error on first turn, return a state with error
                 error_state = BotState(user_input=user_input, text_content=TextContent(text=f"An error occurred: {e}"))
                 return error_state


