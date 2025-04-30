
import json
from typing import Any
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from scratchpad_route_state import RouterState
from scratchpad_tools import TOOL_FUNCTIONS, unknown_intent

class ScratchPadRouter:
    def __init__(self, llm_router: AzureChatOpenAI):
        self.llm_router = llm_router

    def route(self, state: RouterState) -> RouterState:
        # Compose prompt with examples for improved few-shot intent classification
        examples = [
            {"user": "What all apis are there?", "tool": "list_apis"},
            {"user": "How to create a product?", "tool": "generate_sequence"},
            {"user": "What is payload for create?", "tool": "generate_payloads"},
            {"user": "Generate API execution graph.", "tool": "generate_sequence"},
            {"user": "Add getAll at start.", "tool": "add_edge"},
            {"user": "Verify created product after update.", "tool": "verify_created"},
            {"user": "Explain execution graph.", "tool": "explain_graph"},
            {"user": "Explain create product api.", "tool": "explain_endpoint"}
        ]
        # Build few-shot prompt
        prompt = """
You are an OpenAPI assistant. Choose the appropriate tool for the user request.
Available tools: {}.
Respond with exactly the tool name or JSON for a plan.
Examples:
""".format(list(TOOL_FUNCTIONS.keys()))
        for ex in examples:
            prompt += f"User: {ex['user']}\nAssistant: {ex['tool']}\n"
        prompt += f"User: {state.user_input}\nAssistant:"

        # LLM call
        resp = self.llm_router([HumanMessage(content=prompt)]).content.strip()
        # parse plan or single tool
        try:
            out = json.loads(resp)
            state.plan = out.get("plan", [])
            state.next_step = "execute_plan"
        except json.JSONDecodeError:
            state.next_step = resp if resp in TOOL_FUNCTIONS else "unknown_intent"

        # Execute
        if state.next_step == "execute_plan":
            for step in state.plan:
                fn = TOOL_FUNCTIONS.get(step, unknown_intent)
                state = fn(state)
        else:
            state = TOOL_FUNCTIONS.get(state.next_step, unknown_intent)(state)
        return state

    def build(self) -> Any:
        builder = StateGraph(RouterState)
        builder.add_node("route", self.route)
        for name, fn in TOOL_FUNCTIONS.items():
            builder.add_node(name, fn)
        builder.add_conditional_edges("route", lambda s: s.next_step or "unknown_intent", {n: n for n in TOOL_FUNCTIONS})
        builder.add_edge(START, "route")
        builder.add_edge("route", END)
        return builder.compile()
