# llm_router_graph.py
# state_schema.py
from typing import List, Optional
from pydantic import BaseModel
from api_execution_graph_schema import GraphOutput

class RouterState(BaseModel):
    user_input: Optional[str] = None
    next_step: Optional[str] = None
    plan: Optional[List[str]] = None
    openapi_schema: Optional[dict] = None
    execution_graph: Optional[GraphOutput] = None
    payloads: Optional[dict] = None
    response: Optional[str] = None



import os, json
from typing import Any, Dict
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state_schema import RouterState
from api_graph_manager import ApiGraphManager

class LangGraphOpenApiRouter:
    def __init__(self, spec_text: str, llm_router: AzureChatOpenAI, llm_worker: AzureChatOpenAI, cache_dir: str = "./cache"):
        os.makedirs(cache_dir, exist_ok=True)
        ApiGraphManager.set_llm(llm_worker, spec_text)
        self.llm_router = llm_router
        self.cache_dir = cache_dir
        self.tools = {}
        self._register_tools()

    def tool(self, name: str):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco

    def _register_tools(self):
        @self.tool("parse_openapi")
        def parse_openapi(state: Dict[str, Any]):
            path = os.path.join(self.cache_dir, "schema.json")
            if os.path.exists(path):
                state["openapi_schema"] = json.load(open(path))
            else:
                out = ApiGraphManager.generate_api_execution_graph_fn(state["user_input"] or "")
                state["openapi_schema"] = out.graph.model_dump()
                json.dump(state["openapi_schema"], open(path, "w"))
            return state

        @self.tool("generate_sequence")
        def generate_sequence(state):
            out = ApiGraphManager.generate_api_execution_graph_fn(state["user_input"] or "")
            state["execution_graph"] = out.graph
            return state

        @self.tool("generate_payloads")
        def generate_payloads(state):
            path = os.path.join(self.cache_dir, "payloads.json")
            if os.path.exists(path):
                state["payloads"] = json.load(open(path))
            else:
                out = ApiGraphManager.generate_payload_fn(state["user_input"] or "")
                state["payloads"] = out.textContent
                json.dump(state["payloads"], open(path, "w"))
            return state

        @self.tool("answer_openapi")
        def answer_openapi(state):
            out = ApiGraphManager.openapi_help_fn(state["user_input"] or "")
            state["response"] = out.textContent
            return state

        @self.tool("simulate_load_test")
        def simulate_load_test(state):
            n = next((int(w) for w in (state["user_input"] or "").split() if w.isdigit()), 1)
            out = ApiGraphManager.simulate_load_test_fn(num_users=n)
            state["response"] = out.textContent
            return state

        @self.tool("execute_workflow")
        def execute_workflow(state):
            out = ApiGraphManager.execute_workflow_fn(state["user_input"] or "")
            state["response"] = out.textContent
            return state

        @self.tool("unknown_intent")
        def unknown_intent(state):
            state["response"] = "Sorry, I didn't understand that."
            return state

        @self.tool("execute_plan")
        def execute_plan(state):
            for step in state.get("plan", []):
                state = self.tools.get(step, unknown_intent)(state)
            return state

    def route_intent(self, state: Dict[str, Any]):
        prompt = (
            "You are an OpenAPI assistant.\n"
            f"Choose one action from {list(self.tools.keys())}.\n"
            f"User Input: {state['user_input']}"
        )
        resp = self.llm_router([HumanMessage(content=prompt)])
        try:
            j = json.loads(resp.content)
            state["next_step"] = j["next_step"]
            if "plan" in j: state["plan"] = j["plan"]
        except:
            state["next_step"] = resp.content.strip()
        return state

    def build_graph(self):
        builder = StateGraph(RouterState)

        # entry
        builder.add_node("route_intent", self.route_intent)
        builder.add_edge(START, "route_intent")

        # tool nodes
        for name, fn in self.tools.items():
            builder.add_node(name, fn)

        # conditional routing
        builder.add_conditional_edges(
            "route_intent",
            lambda s: s.next_step or "unknown_intent",
            {k: k for k in self.tools}
        )

        # static multi-step: payload → sequence
        builder.add_edge("generate_payloads", "generate_sequence")

        # finish node
        def finish(state): return state
        builder.add_node("finish", finish)
        for terminal in ["generate_payloads","generate_sequence","execute_plan","answer_openapi","simulate_load_test","execute_workflow","unknown_intent"]:
            builder.add_edge(terminal, "finish")
        builder.add_edge(START, "finish", )  # ensure no orphan finish

        builder.set_entry_point(START)
        builder.set_finish_point("finish")

        return builder.compile()

if __name__ == "__main__":
    spec = open("petstore.yaml").read()
    router_llm = AzureChatOpenAI(deployment_name="gpt-35-router", temperature=0)
    worker_llm = AzureChatOpenAI(deployment_name="gpt-35-worker", temperature=0)
    router = LangGraphOpenApiRouter(spec, router_llm, worker_llm)
    graph = router.build_graph()
    while True:
        ui = input("You: ")
        out = graph.invoke({"user_input": ui})
        print("Assistant:", out.response)
