import os
import json
from typing import Any, Dict, List

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
        self.spec_text = spec_text
        self.llm_router = llm_router
        self.llm_worker = llm_worker
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.api_manager = ApiGraphManager(llm=self.llm_worker, openapi_yaml=self.spec_text)
        self.memory = MemorySaver()

    def route_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are an OpenAPI assistant.\n"
            "Based on the user input, choose one or more actions from:\n"
            "[parse_openapi, generate_sequence, generate_payloads, answer_openapi, simulate_load_test, execute_workflow].\n"
            "Respond with a JSON object: either {\"next_step\": <step>} or {\"plan\": [steps]}.\n"
            f"User Input: {state['user_input']}"
        )
        resp = self.llm_router([HumanMessage(content=prompt)])
        text = resp.content.strip()
        try:
            data = json.loads(text)
            if "plan" in data:
                state["plan"] = data["plan"]
                state["next_step"] = "execute_plan"
            else:
                state["next_step"] = data.get("next_step")
        except json.JSONDecodeError:
            state["next_step"] = text
        return state

    # tool wrappers
    def parse_openapi(self, state): return self.api_manager.parse_spec_fn(state)
    def generate_sequence(self, state): return self.api_manager.generate_api_execution_graph_fn(state)
    def generate_payloads(self, state): return self.api_manager.generate_payload_fn(state)
    def answer_openapi(self, state): return self.api_manager.openapi_help_fn(state)
    def simulate_load_test(self, state): return self.api_manager.simulate_load_test_fn(state)
    def execute_workflow(self, state): return self.api_manager.execute_workflow_fn(state)

    # plan executor
    def execute_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        for step in state.get("plan", []):
            fn = getattr(self, step, None)
            if fn:
                state = fn(state)
            else:
                state["response"] = f"Unknown step: {step}"
                break
        return state

    def unknown_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["response"] = "Sorry, I didn't understand that."
        return state

    def build_graph(self) -> StateGraph:
        builder = StateGraph()
        builder.add_node("route_intent", self.route_intent)
        builder.set_entry_point("route_intent")
        for step in ["parse_openapi","generate_sequence","generate_payloads","answer_openapi","simulate_load_test","execute_workflow","execute_plan","unknown_intent"]:
            builder.add_node(step, getattr(self, step))
        builder.add_conditional_edges(
            "route_intent",
            lambda s: s.get("next_step", "unknown_intent"),
            {step: step for step in ["parse_openapi","generate_sequence","generate_payloads","answer_openapi","simulate_load_test","execute_workflow","execute_plan","unknown_intent"]}
        )
        builder.set_finish_point("*")
        return builder.compile()

if __name__ == "__main__":
    spec = open(os.getenv("OPENAPI_SPEC","petstore.yaml")).read()
    router_llm = AzureChatOpenAI(deployment_name="gpt-35-router", temperature=0)
    worker_llm = AzureChatOpenAI(deployment_name="gpt-35-worker", temperature=0)
    router = LangGraphOpenApiRouter(spec, router_llm, worker_llm)
    graph = router.build_graph()
    while True:
        inp = input("You: ")
        out = graph.invoke({"user_input":inp})
        print("Assistant:", out.get("response",""))
```
