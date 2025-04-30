# router.py
import json
# Assume langchain.schema and langchain_openai are available
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI # Using MockLLM from tools.py
from langgraph.graph import StateGraph, START, END
from models import BotState, BotIntent, AVAILABLE_TOOLS # Use BotState as the graph state
from tools import TOOL_FUNCTIONS # Import tool functions mapping
# We will pass LLM instances from main.py, so no global llm here

class BotRouter:
    """
    Orchestrates the bot's workflow using LangGraph.
    Uses a dedicated router LLM to decide the next tool (node) to execute.
    """
    def __init__(self, llm_router: Any): # Accept the router LLM instance
        self.llm_router = llm_router

    def route_step(self, state: BotState) -> str:
        """
        This node determines the next tool/intent based on user input and history
        using the router LLM. It returns the string name (value) of the next BotIntent.
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
        # Check if the LLM's response is a valid tool name defined in TOOL_FUNCTIONS
        next_step_name = res if res in TOOL_FUNCTIONS else BotIntent.UNKNOWN.value

        # Update state with the chosen intent
        try:
            # Validate the chosen name against the BotIntent Enum
            state.intent = BotIntent(next_step_name)
        except ValueError:
            # Fallback if LLM output is not a valid enum value (should be caught by `res in TOOL_FUNCTIONS` but double-checking)
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

    def build_graph(self, llm_worker: Any):
        """
        Build the LangGraph StateGraph.
        Accepts the worker LLM instance to pass to the tool nodes.
        """
        # The graph state is our Pydantic BotState model
        builder = StateGraph(BotState)

        # Add the router node
        builder.add_node("route", self.route_step) # This node uses self.llm_router

        # Add nodes for each tool function
        # Each tool function receives and returns the BotState
        # We wrap the tool function in a lambda to pass the worker LLM instance
        tool_nodes = {name: lambda state, f=func: f(state, llm_worker) for name, func in TOOL_FUNCTIONS.items()}
        # If you had multiple worker LLMs (e.g., llm_parser, llm_generator),
        # you would modify the lambda for specific tools:
        # tool_nodes[BotIntent.OPENAPI_HELP.value] = lambda state: openapi_help(state, llm_parser)
        # tool_nodes[BotIntent.GENERATE_GRAPH.value] = lambda state: generate_api_execution_graph(state, llm_generator)
        # etc. For this example, one llm_worker is used for all tools.


        for name, node in tool_nodes.items():
             builder.add_node(name, node)

        # Define the entry point of the graph
        builder.add_edge(START, "route")

        # Define the transitions from the router node
        # The conditional edge uses the return value of the 'route_step' node
        # which is the string name of the chosen tool.
        # The mapping {name: name for name in TOOL_FUNCTIONS.keys()}
        # means if route_step returns "tool_name", the graph transitions to the node named "tool_name".
        builder.add_conditional_edges(
            "route",
            lambda next_step_name: next_step_name, # The next state key is the return value of route_step
            # Map the string name returned by route_step to the corresponding node name
            # Ensure this mapping covers all possible string outputs of route_step
            TOOL_FUNCTIONS # Directly use the TOOL_FUNCTIONS dict as the mapping
        )

        # Define transitions from the tool nodes
        # Most tools should transition back to the router for the next turn,
        # allowing the bot to process follow-up questions or actions.
        # Some tools (like unknown or get_graph_json) might end the conversation turn.
        for tool_name in TOOL_FUNCTIONS.keys():
             if tool_name == BotIntent.UNKNOWN.value:
                  # If intent is unknown, end the current turn.
                  builder.add_edge(tool_name, END)
             elif tool_name == BotIntent.GET_GRAPH_JSON.value:
                  # Outputting JSON is likely the final action for that specific request.
                  builder.add_edge(tool_name, END)
             elif tool_name == BotIntent.GENERAL_QUERY.value:
                 # A general query might be a one-off question, so end the turn.
                 builder.add_edge(tool_name, END)
             # Note: OPENAPI_HELP now goes back to route to allow follow-up after parsing/help
             # elif tool_name == BotIntent.OPENAPI_HELP.value:
             #     builder.add_edge(tool_name, END) # Example: end after help

             else:
                  # Most other tools (generate_graph, add_edge, describe_graph, generate_payload, openapi_help)
                  # likely require further interaction or explanation, so go back to the router.
                  builder.add_edge(tool_name, "route")

        # Compile the graph
        app = builder.compile()
        return app

