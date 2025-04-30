import json
from langchain.schema import HumanMessage  # for LLM calls :contentReference[oaicite:0]{index=0}
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END  # LangGraph usage :contentReference[oaicite:1]{index=1}
from models import RouterState
from scratchpad_tools import TOOL_FUNCTIONS

class ScratchPadRouter:
    def __init__(self, llm_router: AzureChatOpenAI):
        self.llm_router = llm_router

    def route(self, state: RouterState) -> RouterState:
        # Construct prompt with scratchpad history
        history = state.scratchpad.get("dialogue", "")
        prompt = (
            f"{history}\nUser: {state.user_input}\nAssistant (choose intent/tool):"
        )
        resp = self.llm_router([HumanMessage(content=prompt)]).content
        state.intent = resp.strip()
        # update scratchpad dialogue
        state.scratchpad["dialogue"] = prompt + "\n" + resp
        # If plan returned as JSON, parse it
        try:
            out = json.loads(resp)
            state.plan = out.get("plan", [])
            state.next_step = "execute_plan"
        except json.JSONDecodeError:
            state.next_step = state.intent if state.intent in TOOL_FUNCTIONS else "unknown_intent"

        # Execute next step or plan
        if state.next_step == "execute_plan":
            for step in state.plan:
                state = TOOL_FUNCTIONS.get(step, unknown_intent)(state)
        else:
            state = TOOL_FUNCTIONS[state.next_step](state)
        return state

    def build(self) -> Any:
        builder = StateGraph(RouterState)  # schema as TypedDict :contentReference[oaicite:2]{index=2}
        builder.add_node("route", self.route)
        for name, fn in TOOL_FUNCTIONS.items():
            builder.add_node(name, fn)
        builder.add_conditional_edges("route", lambda s: s.next_step, {n: n for n in TOOL_FUNCTIONS})
        builder.add_edge(START, "route")
        builder.add_edge("route", END)
        return builder.compile()
