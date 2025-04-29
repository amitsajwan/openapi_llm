
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

# ── Logging Setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)  # root logger :contentReference[oaicite:5]{index=5}
logger = logging.getLogger(__name__)

class LangGraphOpenApiRouter:
    def __init__(
        self, spec_text: str,
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
        logger.info("Tools registered: %s", list(self.tools.keys()))

    def tool(self, name: str):
        def decorator(fn):
            self.tools[name] = fn
            logger.info("Registered tool node: %s", name)
            return fn
        return decorator

    def _register_tools(self):
        @self.tool("parse_openapi")
        def parse_openapi(state):
            logger.info("Node ▶ parse_openapi")
            path = os.path.join(self.cache_dir, "schema.json")
            if os.path.exists(path):
                state["openapi_schema"] = json.load(open(path))
                logger.info("Loaded cached schema")
            else:
                out = ApiGraphManager.generate_api_execution_graph_fn(state["user_input"] or "")
                state["openapi_schema"] = out.graph.model_dump()
                json.dump(state["openapi_schema"], open(path,"w"))
                logger.info("Parsed & cached schema")
            return state

        @self.tool("generate_sequence")
        def generate_sequence(state):
            logger.info("Node ▶ generate_sequence")
            out = ApiGraphManager.generate_api_execution_graph_fn(state["user_input"] or "")
            state["execution_graph"] = out.graph
            return state

        @self.tool("generate_payloads")
        def generate_payloads(state):
            logger.info("Node ▶ generate_payloads")
            path = os.path.join(self.cache_dir, "payloads.json")
            if os.path.exists(path):
                state["payloads"] = json.load(open(path))
                logger.info("Loaded cached payloads")
            else:
                out = ApiGraphManager.generate_payload_fn(state["user_input"] or "")
                state["payloads"] = out.textContent
                json.dump(state["payloads"], open(path,"w"))
                logger.info("Generated & cached payloads")
            return state

        @self.tool("answer_openapi")
        def answer_openapi(state):
            logger.info("Node ▶ answer_openapi")
            out = ApiGraphManager.openapi_help_fn(state["user_input"] or "")
            state["response"] = out.textContent
            return state

        @self.tool("simulate_load_test")
        def simulate_load_test(state):
            logger.info("Node ▶ simulate_load_test")
            n = next((int(w) for w in (state["user_input"] or "").split() if w.isdigit()), 1)
            out = ApiGraphManager.simulate_load_test_fn(num_users=n)
            state["response"] = out.textContent
            return state

        @self.tool("execute_workflow")
        def execute_workflow(state):
            logger.info("Node ▶ execute_workflow")
            out = ApiGraphManager.execute_workflow_fn(state["user_input"] or "")
            state["response"] = out.textContent
            return state

        @self.tool("execute_plan")
        def execute_plan(state):
            logger.info("Node ▶ execute_plan; plan=%s", state.get("plan"))
            for step in state.get("plan", []):
                logger.info(" → plan step: %s", step)
                state = self.tools.get(step, lambda s: s)(state)
            return state

        @self.tool("unknown_intent")
        def unknown_intent(state):
            logger.info("Node ▶ unknown_intent")
            state["response"] = "Sorry, I didn't understand that."
            return state

    def route_intent(self, state):
        logger.info("Node ▶ route_intent; user_input=%r", state["user_input"])
        prompt = (
            "You are an OpenAPI assistant.\n"
            f"Choose exactly one action from {list(self.tools.keys())}.\n"
            f"User Input: {state['user_input']}\n"
            "Or return JSON {\"next_step\":\"…\",\"plan\":[…]} for multi-step."
        )
        resp = self.llm_router([HumanMessage(content=prompt)])
        try:
            j = json.loads(resp.content)
            state["next_step"] = j["next_step"]
            state["plan"] = j.get("plan")
            logger.info("Router ▶ structured: %s", j)
        except:
            state["next_step"] = resp.content.strip()
            logger.info("Router ▶ single-step: %s", state["next_step"])
        return state

    def build_graph(self):
        builder = StateGraph(RouterState)  # supply schema :contentReference[oaicite:6]{index=6}

        # ENTRY: START → route_intent :contentReference[oaicite:7]{index=7}
        logger.info("ADDING EDGE: %s → route_intent", START)
        builder.add_edge(START, "route_intent")
        builder.set_entry_point("route_intent")  # real node, not START

        # Add all tool nodes :contentReference[oaicite:8]{index=8}
        for name, fn in [("route_intent", self.route_intent)] + list(self.tools.items()):
            logger.info("ADDING NODE: %s", name)
            builder.add_node(name, fn)

        # Conditional routing :contentReference[oaicite:9]{index=9}
        logger.info("ADDING CONDITIONAL EDGES from route_intent")
        builder.add_conditional_edges(
            "route_intent",
            lambda s: s.next_step or "unknown_intent",
            {k: k for k in self.tools}
        )

        # Static chain example :contentReference[oaicite:10]{index=10}
        logger.info("ADDING EDGE: generate_payloads → generate_sequence")
        builder.add_edge("generate_payloads", "generate_sequence")

        # FINISH: real node → END :contentReference[oaicite:11]{index=11}
        def finish(state): 
            logger.info("Reached finish node") 
            return state

        logger.info("ADDING NODE: finish")
        builder.add_node("finish", finish)
        for t in self.tools:
            logger.info("ADDING EDGE: %s → finish", t)
            builder.add_edge(t, "finish")
        logger.info("ADDING EDGE: finish → %s", END)
        builder.add_edge("finish", END)
        builder.set_finish_point("finish")

        graph = builder.compile()  # must compile :contentReference[oaicite:12]{index=12}
        logger.info("COMPILED NODES: %s", graph.nodes)
        logger.info("COMPILED EDGES: %s", graph.edges)
        return graph

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
