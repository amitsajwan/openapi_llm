import os, json
from typing import Any, Dict
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from graph_tools import APIGraphManager
from models import RouterState

class OpenApiReactRouterManager:
    def __init__(self, spec_text: str, llm_router: AzureChatOpenAI, llm_worker: AzureChatOpenAI, cache_dir: str = "./cache"):
        self.spec_text = spec_text
        self.llm_router = llm_router
        self.llm_worker = llm_worker
        self.cache_dir = cache_dir
        self.api_graph_manager = APIGraphManager.set_llms(llm_worker)
        self.spec = self.api_graph_manager.getPrompts(spec_text)
        self.tools = {}
        self.register_tools()

    def tool(self, name: str):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco

    def register_tools(self):
        @self.tool("parse_openapi")
        def parse_openapi(state: RouterState) -> RouterState:
            state.openapi_schema = self.spec_text
            state.scratchpad += "Parsed OpenAPI schema.\n"
            return state

        @self.tool("generate_sequence")
        def generate_sequence(state: RouterState) -> RouterState:
            out = self.api_graph_manager.generate_api_execution_graph_fn(state.user_input or "")
            state.execution_graph = out.graph
            state.response = out.textContent
            state.scratchpad += "Generated API execution sequence.\n"
            return state

        @self.tool("generate_payloads")
        def generate_payloads(state: RouterState) -> RouterState:
            path = os.path.join(self.cache_dir, "payloads.json")
            if os.path.exists(path):
                with open(path) as f:
                    state.payloads = json.load(f)
                    state.response = f"Loaded from cache: {path}"
                    state.scratchpad += "Loaded payloads from cache.\n"
            else:
                out = self.api_graph_manager.generate_payloads_fn(state.user_input or "")
                state.payloads = out.payloads
                state.response = out.textContent
                with open(path, "w") as f:
                    json.dump(state.payloads, f)
                state.scratchpad += "Generated payloads and saved to cache.\n"
            return state

        @self.tool("answer")
        def answer(state: RouterState) -> RouterState:
            out = self.api_graph_manager.openapi_help_fn(state.user_input or "")
            state.response = out.textContent
            state.scratchpad += "Answered user question.\n"
            return state

        @self.tool("simulate_load_test")
        def simulate_load_test(state: RouterState) -> RouterState:
            n = next((int(w) for w in (state.user_input or "").split() if w.isdigit()), 1)
            out = self.api_graph_manager.simulate_load_test_fn(n)
            state.response = out.textContent
            state.scratchpad += f"Simulated load test with {n} users.\n"
            return state

        @self.tool("execute_workflow")
        def execute_workflow(state: RouterState) -> RouterState:
            out = self.api_graph_manager.execute_workflow_fn(state)
            state.response = out.textContent
            state.scratchpad += "Executed workflow.\n"
            return state

        @self.tool("execute_plan")
        def execute_plan(state: RouterState) -> RouterState:
            state.response = ""
            for step in state.plan or []:
                if step in self.tools:
                    state = self.tools[step](state)
            state.scratchpad += "Executed plan steps.\n"
            return state

        @self.tool("unknown_intent")
        def unknown_intent(state: RouterState) -> RouterState:
            state.response = "Sorry, I didn’t understand that."
            state.scratchpad += "Unknown user intent.\n"
            return state

        @self.tool("add_edge")
        def add_edge(state: RouterState) -> RouterState:
            self.api_graph_manager.add_edge_fn(state.user_input or "", state.execution_graph)
            state.response = f"Edge added to execution graph"
            state.scratchpad += "Added edge to execution graph.\n"
            return state

        @self.tool("validate_graph")
        def validate_graph(state: RouterState) -> RouterState:
            graph = state.execution_graph
            is_valid, reason = self.validate_graph_dict(graph)
            state.response = f"Execution graph is valid: {is_valid}. Reason: {reason}"
            state.scratchpad += f"Validated graph: {is_valid}, reason: {reason}.\n"
            return state

    def validate_graph_dict(self, data: Dict) -> tuple[bool, str]:
        import networkx as nx
        try:
            G = nx.DiGraph()
            for node, nxt in data.items():
                for n in nxt.get("next", []):
                    G.add_edge(node, n)
            if not nx.is_directed_acyclic_graph(G):
                return False, "Graph contains cycles."
            return True, "Graph is valid."
        except Exception as e:
            return False, f"Validation failed: {e}"

def route_intent(self, state: RouterState) -> RouterState:
    # Update context
    preamble = (
        "You are OpenAPI assistant router.\n"
        f"Current graph/API status:\n{state.user_input}\n"
        f"Execution graph: {'yes' if state.execution_graph else 'no'}\n"
        f"Payloads generated: {'yes' if state.payloads else 'no'}\n"
        f"Available tools: {list(self.tools.keys())}\n"
    )

    routing_instructions = (
        "If the user asks a question, use 'answer'. "
        "If they want to run something, use 'execute_workflow' or 'simulate_load_test'. "
        "If they want to build/modify sequence, use 'generate_sequence'. "
        "No need to add edge when we generate the graph from scratch. "
        "If they ask to generate payloads, use 'generate_payloads'. "
        "If a multi-step process is needed, output a plan (e.g., JSON: {\"next_step\": [\"tool1\", \"tool2\"]})."
    )

    # Include the scratchpad (past messages) to give the LLM context
    scratchpad_text = state.scratchpad if state.scratchpad else ""
    prompt = (
        f"{preamble}\n"
        f"{routing_instructions}\n\n"
        f"{scratchpad_text}\n"
        f"User: {state.user_input}\nAssistant:"
    )

    # LLM call
    response = self.llm_router([HumanMessage(content=prompt)]).content.strip()
    state.intent = response

    # Update scratchpad
    state.scratchpad = (state.scratchpad or "") + f"\nUser: {state.user_input}\nAssistant: {response}"

    # Try to parse plan or next_step
    try:
        output = json.loads(response)
        if "plan" in output:
            state.plan = output["plan"]
            state.next_step = "execute_plan"
        elif "next_step" in output and output["next_step"] in self.tools:
            state.next_step = output["next_step"]
        else:
            state.next_step = "unknown_intent"
            state.plan = None
    except json.JSONDecodeError:
        state.next_step = "unknown_intent"
        state.plan = None

    # Save action history
    if state.user_input:
        state.action_history.append((state.user_input, state.next_step))

    return state

    def build_graph(self):
        builder = StateGraph(RouterState)
        builder.add_node("route_intent", self.route_intent)
        for name, fn in self.tools.items():
            builder.add_node(name, fn)

        builder.add_conditional_edges("route_intent", lambda s: s.next_step or "unknown_intent", {
            k: k for k in self.tools
        } | {
            "unknown_intent": "unknown_intent"
        })

        builder.add_edge("generate_payloads", "generate_sequence")
        builder.add_edge("generate_sequence", "execute_workflow")

        builder.add_node("finish", lambda x: x)
        builder.add_edge("execute_plan", "finish")

        builder.set_entry_point("route_intent")
        builder.set_finish_point("finish")

        return builder.compile()
