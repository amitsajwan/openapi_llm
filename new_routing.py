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

# llm_router_graph.py

import os
import json
import logging
from typing import Any, Dict

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state_schema import RouterState
from api_graph_manager import ApiGraphManager

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)  # module-level logger :contentReference[oaicite:2]{index=2}

class LangGraphOpenApiRouter:
    def __init__(
        self,
        spec_text: str,
        llm_router: AzureChatOpenAI,
        llm_worker: AzureChatOpenAI,
        cache_dir: str = "./cache"
    ):
        os.makedirs(cache_dir, exist_ok=True)
        ApiGraphManager.set_llm(llm_worker, spec_text)
        self.llm_router = llm_router
        self.cache_dir = cache_dir
        self.tools: Dict[str, Any] = {}
        self._register_tools()
        logger.info("Initialized LangGraphOpenApiRouter with tools: %s", list(self.tools.keys()))

    def tool(self, name: str):
        def deco(fn):
            self.tools[name] = fn
            logger.info("Registered tool node: %s", name)
            return fn
        return deco

    def _register_tools(self):
        @self.tool("parse_openapi")
        def parse_openapi(state):
            logger.info("Executing node: parse_openapi")
            path = os.path.join(self.cache_dir, "schema.json")
            if os.path.exists(path):
                state["openapi_schema"] = json.load(open(path))
                logger.info("Loaded cached schema from %s", path)
            else:
                out = ApiGraphManager.generate_api_execution_graph_fn(state["user_input"] or "")
                state["openapi_schema"] = out.graph.model_dump()
                json.dump(state["openapi_schema"], open(path, "w"))
                logger.info("Parsed and cached schema to %s", path)
            return state

        @self.tool("generate_sequence")
        def generate_sequence(state):
            logger.info("Executing node: generate_sequence")
            out = ApiGraphManager.generate_api_execution_graph_fn(state["user_input"] or "")
            state["execution_graph"] = out.graph
            return state

        @self.tool("generate_payloads")
        def generate_payloads(state):
            logger.info("Executing node: generate_payloads")
            path = os.path.join(self.cache_dir, "payloads.json")
            if os.path.exists(path):
                state["payloads"] = json.load(open(path))
                logger.info("Loaded cached payloads from %s", path)
            else:
                out = ApiGraphManager.generate_payload_fn(state["user_input"] or "")
                state["payloads"] = out.textContent
                json.dump(state["payloads"], open(path, "w"))
                logger.info("Generated and cached payloads to %s", path)
            return state

        @self.tool("answer_openapi")
        def answer_openapi(state):
            logger.info("Executing node: answer_openapi")
            out = ApiGraphManager.openapi_help_fn(state["user_input"] or "")
            state["response"] = out.textContent
            return state

        @self.tool("simulate_load_test")
        def simulate_load_test(state):
            logger.info("Executing node: simulate_load_test")
            n = next((int(w) for w in (state["user_input"] or "").split() if w.isdigit()), 1)
            out = ApiGraphManager.simulate_load_test_fn(num_users=n)
            state["response"] = out.textContent
            return state

        @self.tool("execute_workflow")
        def execute_workflow(state):
            logger.info("Executing node: execute_workflow")
            out = ApiGraphManager.execute_workflow_fn(state["user_input"] or "")
            state["response"] = out.textContent
            return state

        @self.tool("execute_plan")
        def execute_plan(state):
            logger.info("Executing node: execute_plan with plan: %s", state.get("plan"))
            for step in state.get("plan", []):
                logger.info("Plan step -> %s", step)
                state = self.tools.get(step, lambda s: s)(state)
            return state

        @self.tool("unknown_intent")
        def unknown_intent(state):
            logger.info("Executing node: unknown_intent")
            state["response"] = "Sorry, I didn't understand that."
            return state

    def route_intent(self, state):
        logger.info("Executing node: route_intent; user_input=%r", state["user_input"])
        prompt = (
            "You are an OpenAPI assistant.\n"
            f"Choose one action from {list(self.tools.keys())}.\n"
            f"User Input: {state['user_input']}"
        )
        resp = self.llm_router([HumanMessage(content=prompt)])
        try:
            j = json.loads(resp.content)
            state["next_step"] = j["next_step"]
            if "plan" in j:
                state["plan"] = j["plan"]
            logger.info("Router returned structured plan: %s", j)
        except json.JSONDecodeError:
            state["next_step"] = resp.content.strip()
            logger.info("Router returned single step: %s", state["next_step"])
        return state

    def build_graph(self):
        # 1) Create the graph with Pydantic schema :contentReference[oaicite:3]{index=3}
        builder = StateGraph(RouterState)

        # 2) Entry edge :contentReference[oaicite:4]{index=4}
        logger.info("Adding edge: START -> route_intent")
        builder.add_edge(START, "route_intent")

        # 3) Add nodes :contentReference[oaicite:5]{index=5}
        builder.add_node("route_intent", self.route_intent)
        for name, fn in self.tools.items():
            logger.info("Adding node: %s", name)
            builder.add_node(name, fn)

        # 4) Conditional edges :contentReference[oaicite:6]{index=6}
        logger.info("Adding conditional edges from route_intent")
        builder.add_conditional_edges(
            "route_intent",
            lambda s: s.next_step or "unknown_intent",
            {k: k for k in self.tools}
        )

        # 5) Static chain: payload -> sequence
        logger.info("Adding static edge: generate_payloads -> generate_sequence")
        builder.add_edge("generate_payloads", "generate_sequence")

        # 6) Finish node and edges :contentReference[oaicite:7]{index=7}
        logger.info("Adding END node and edges from terminal nodes")
        def end_node(state): 
            logger.info("Reached END node") 
            return state
        builder.add_node(END, end_node)
        for terminal in ["generate_payloads","generate_sequence","execute_plan","answer_openapi","simulate_load_test","execute_workflow","unknown_intent"]:
            logger.info("Adding edge: %s -> END", terminal)
            builder.add_edge(terminal, END)

        # 7) Declare entry & finish :contentReference[oaicite:8]{index=8}
        builder.set_entry_point(START)
        builder.set_finish_point(END)

        # 8) Log full graph topology
        compiled = builder.compile()
        logger.info("Compiled graph nodes: %s", compiled.nodes)
        logger.info("Compiled graph edges: %s", compiled.edges)
        return compiled

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
