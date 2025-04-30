# scratchpad_route_state.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RouterState:
    # Raw spec text for LLM-driven parsing
    openapi_spec_text: Optional[str] = None
    # Parsed schema cached in scratchpad
    openapi_schema: Optional[Any] = None
    # Execution graph (DAG) cached in scratchpad
    execution_graph: Optional[Dict[str, Any]] = None
    # Generated payloads cached in scratchpad
    payloads: Optional[Dict[str, Any]] = None
    # Bot responses
    response: Optional[str] = None
    # LLM-chosen intent or tool name
    intent: Optional[str] = None
    # Next tool step
    next_step: Optional[str] = None
    # Plan of multiple steps (tool names)
    plan: List[str] = field(default_factory=list)
    # History of (user_input, intent)
    action_history: List[tuple] = field(default_factory=list)
    # Scratchpad storing intermediate LLM reasoning, cached results
    scratchpad: Dict[str, Any] = field(default_factory=dict)


# scratchpad_tools.py
import json
from typing import Dict, Any
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from models import RouterState

# All tools will use LLM for logic, appending reasoning to scratchpad

def parse_openapi(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    # LLM-driven parsing explanation
    prompt = f"Parse this OpenAPI spec into a JSON schema object: {state.openapi_spec_text}"
    res = llm([HumanMessage(content=prompt)]).content
    schema = json.loads(res)
    state.openapi_schema = schema
    state.scratchpad['openapi_schema'] = schema
    state.scratchpad['reason'] = state.scratchpad.get('reason', '') + f"Parsed schema: {res}\n"
    state.response = "OpenAPI schema parsed."
    return state


def generate_sequence(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    prompt = f"Given this OpenAPI schema JSON: {json.dumps(state.openapi_schema)} generate an execution graph as JSON DAG of operationIds with dependencies."
    res = llm([HumanMessage(content=prompt)]).content
    graph = json.loads(res)
    state.execution_graph = graph
    state.scratchpad['execution_graph'] = graph
    state.scratchpad['reason'] += f"Generated execution graph: {res}\n"
    state.response = "Execution graph generated."
    return state


def generate_payloads(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    prompt = f"For each operation in this OpenAPI schema JSON: {json.dumps(state.openapi_schema)}, generate example JSON payloads. Return map of operationId to payload."
    res = llm([HumanMessage(content=prompt)]).content
    payloads = json.loads(res)
    state.payloads = payloads
    state.scratchpad['payloads'] = payloads
    state.scratchpad['reason'] += f"Generated payloads: {res}\n"
    state.response = "Payloads generated."
    return state


def answer(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    prompt = f"Answer based on this OpenAPI schema JSON: {json.dumps(state.openapi_schema)}. Question: {state.user_input}"
    res = llm([HumanMessage(content=prompt)]).content
    state.response = res
    state.scratchpad['reason'] += f"Answered question: {res}\n"
    return state


def add_edge(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    prompt = f"Modify this execution graph JSON: {json.dumps(state.execution_graph)} by {state.user_input}. Return updated graph JSON."
    res = llm([HumanMessage(content=prompt)]).content
    graph = json.loads(res)
    state.execution_graph = graph
    state.scratchpad['execution_graph'] = graph
    state.scratchpad['reason'] += f"Modified graph: {res}\n"
    state.response = "Execution graph updated."
    return state


def validate_graph(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    prompt = f"Check if this DAG JSON: {json.dumps(state.execution_graph)} has cycles. Return JSON { {'valid': bool, 'reason': str} }."  
    res = llm([HumanMessage(content=prompt)]).content
    out = json.loads(res)
    state.response = f"Graph valid: {out['valid']}. Reason: {out['reason']}"
    state.scratchpad['reason'] += f"Validated graph: {res}\n"
    return state


def unknown_intent(state: RouterState, llm: AzureChatOpenAI) -> RouterState:
    state.response = "Sorry, I didn't understand that."
    state.scratchpad['reason'] += "Unknown intent.\n"
    return state


TOOL_FUNCTIONS = {
    "parse_openapi": parse_openapi,
    "generate_sequence": generate_sequence,
    "generate_payloads": generate_payloads,
    "answer": answer,
    "add_edge": add_edge,
    "validate_graph": validate_graph,
    "unknown_intent": unknown_intent,
}


# scratch_pad_routing.py
import json
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from models import RouterState
from scratchpad_tools import TOOL_FUNCTIONS, unknown_intent

class ScratchPadRouter:
    def __init__(self, llm_router: AzureChatOpenAI):
        self.llm_router = llm_router

    def route(self, state: RouterState) -> RouterState:
        # include scratchpad reasoning in prompt
        history = state.scratchpad.get("reason", "")
        prompt = f"{history}\nUser: {state.user_input}\nChoose next action or plan as JSON."
        resp = self.llm_router([HumanMessage(content=prompt)]).content
        # update reasoning scratchpad
        state.scratchpad['reason'] = history + f"LLM routing: {resp}\n"
        # parse JSON plan or intent
        try:
            out = json.loads(resp)
            if 'plan' in out:
                state.plan = out['plan']
                state.next_step = 'execute_plan'
            elif 'next_step' in out:
                state.next_step = out['next_step']
            else:
                state.next_step = 'unknown_intent'
        except json.JSONDecodeError:
            state.next_step = resp.strip() if resp.strip() in TOOL_FUNCTIONS else 'unknown_intent'

        state.intent = state.next_step
        state.action_history.append((state.user_input, state.intent))

        # execute a plan or single step
        if state.next_step == 'execute_plan':
            for step in state.plan:
                fn = TOOL_FUNCTIONS.get(step, unknown_intent)
                state = fn(state, self.llm_router)
        else:
            fn = TOOL_FUNCTIONS.get(state.next_step, unknown_intent)
            state = fn(state, self.llm_router)
        return state

    def build(self):
        builder = StateGraph(RouterState)
        builder.add_node('route', self.route)
        for name, fn in TOOL_FUNCTIONS.items():
            builder.add_node(name, fn)
        builder.add_conditional_edges('route', lambda s: s.next_step or 'unknown_intent', {n: n for n in TOOL_FUNCTIONS})
        builder.add_edge(START, 'route')
        builder.add_edge('route', END)
        return builder.compile()
