import os
import json
from typing import Any, Dict, Callable, List

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from api_graph_manager import ApiGraphManager


class LangGraphOpenApiRouter:
    def __init__(
        self,
        spec_text: str,
        llm_router: AzureChatOpenAI,
        llm_worker: AzureChatOpenAI,
        cache_dir: str = "./cache"
    ):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # configure manager
        ApiGraphManager.set_llm(llm_worker, spec_text)

        # router vs worker LLM
        self.llm_router = llm_router
        self.tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._register_tools()

    def tool(self, name: str):
        def deco(fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
            self.tools[name] = fn
            return fn
        return deco

    def _register_tools(self):
        @self.tool("parse_openapi")
        def parse_openapi(state: Dict[str, Any]) -> Dict[str, Any]:
            path = os.path.join(self.cache_dir, "schema.json")
            if os.path.exists(path):
                state["openapi_schema"] = json.load(open(path))
            else:
                out = ApiGraphManager.generate_api_execution_graph_fn(state.get("user_input",""))
                state["openapi_schema"] = out.graph
                json.dump(out.graph.model_dump(), open(path,"w"))
            return state

        @self.tool("generate_sequence")
        def generate_sequence(state: Dict[str, Any]) -> Dict[str, Any]:
            out = ApiGraphManager.generate_api_execution_graph_fn(state.get("user_input",""))
            state["execution_graph"] = out.graph
            return state

        @self.tool("generate_payloads")
        def generate_payloads(state: Dict[str, Any]) -> Dict[str, Any]:
            path = os.path.join(self.cache_dir, "payloads.json")
            if os.path.exists(path):
                state["payloads"] = json.load(open(path))
            else:
                out = ApiGraphManager.generate_payload_fn(state.get("user_input",""))
                state["payloads"] = out.textContent
                json.dump(state["payloads"], open(path,"w"))
            return state

        @self.tool("answer_openapi")
        def answer_openapi(state: Dict[str, Any]) -> Dict[str, Any]:
            out = ApiGraphManager.openapi_help_fn(state.get("user_input",""))
            state["response"] = out.textContent
            return state

        @self.tool("simulate_load_test")
        def simulate_load_test(state: Dict[str, Any]) -> Dict[str, Any]:
            users = next((int(w) for w in state["user_input"].split() if w.isdigit()), 1)
            out = ApiGraphManager.simulate_load_test_fn(num_users=users)
            state["response"] = out.textContent
            return state

        @self.tool("execute_workflow")
        def execute_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
            out = ApiGraphManager.execute_workflow_fn(state.get("user_input",""))
            state["response"] = out.textContent
            return state

        @self.tool("unknown_intent")
        def unknown_intent(state: Dict[str, Any]) -> Dict[str, Any]:
            state["response"] = "Sorry, I didn't understand that."
            return state

        @self.tool("execute_plan")
        def execute_plan(state: Dict[str, Any]) -> Dict[str, Any]:
            for step in state.get("plan", []):
                state = self.tools.get(step, unknown_intent)(state)
            return state

    def route_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are an OpenAPI assistant.\n"
            "Choose exactly one action from:\n"
            f"{list(self.tools.keys())}\n"
            f"User Input: {state['user_input']}\n"
            "Return only the action name, or a JSON object "
            "with keys \"next_step\" and optional \"plan\"."
        )
        resp = self.llm_router([HumanMessage(content=prompt)])
        content = resp.content.strip()
        try:
            j = json.loads(content)
            state["next_step"] = j["next_step"]
            if "plan" in j:
                state["plan"] = j["plan"]
        except:
            state["next_step"] = content
        return state

    def build_graph(self) -> StateGraph:
        builder = StateGraph()
        builder.add_node("route_intent", self.route_intent)
        builder.set_entry_point("route_intent")

        for name, fn in self.tools.items():
            builder.add_node(name, fn)

        builder.add_conditional_edges(
            "route_intent",
            lambda s: s.get("next_step", "unknown_intent"),
            {k: k for k in self.tools}
        )

        builder.set_finish_point("*")
        return builder.compile()


if __name__ == "__main__":
    # demo
    spec = open("petstore.yaml").read()
    router_llm = AzureChatOpenAI(deployment_name="gpt-35-router", temperature=0)
    worker_llm = AzureChatOpenAI(deployment_name="gpt-35-worker", temperature=0)
    router = LangGraphOpenApiRouter(spec, router_llm, worker_llm)
    graph = router.build_graph()

    while True:
        ui = input("You: ")
        out = graph.invoke({"user_input": ui})
        print("Assistant:", out.get("response",""))
