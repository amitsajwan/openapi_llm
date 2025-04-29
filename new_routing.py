import os
import json
from typing import Any, Dict

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

        self.api_manager = ApiGraphManager(
            llm=self.llm_worker,
            openapi_yaml=self.spec_text
        )

        self.memory = MemorySaver()

    def route_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state["user_input"]
        prompt = (
            "You are an OpenAPI assistant.\n"
            "Based on the user input, choose exactly one action from:"
            " [parse_openapi, generate_sequence, generate_payloads, answer_openapi, simulate_load_test, execute_workflow].\n"
            f"User Input: {user_input}\n"
            "Return only the action name."
        )
        resp = self.llm_router([HumanMessage(content=prompt)])
        action = resp.content.strip()
        state["next_step"] = action
        return state

    def parse_openapi(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cache_path = os.path.join(self.cache_dir, "schema.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                state["openapi_schema"] = json.load(f)
        else:
            parsed = self.api_manager.parse_spec_fn(state.get("user_input", ""))
            state["openapi_schema"] = parsed
            with open(cache_path, "w") as f:
                json.dump(parsed, f)
        return state

    def generate_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.api_manager.generate_api_execution_graph_fn(
            state.get("user_input", "")
        )
        state["execution_graph"] = graph
        return state

    def generate_payloads(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cache_path = os.path.join(self.cache_dir, "payloads.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                state["payloads"] = json.load(f)
        else:
            graph = self.api_manager.generate_payload_fn(state.get("user_input", ""))
            payloads = getattr(graph, "payloads", None)
            state["payloads"] = payloads
            with open(cache_path, "w") as f:
                json.dump(payloads, f)
        return state

    def answer_openapi(self, state: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.api_manager.openapi_help_fn(state.get("user_input", ""))
        state["response"] = getattr(graph, "textContent", "")
        return state

    def simulate_load_test(self, state: Dict[str, Any]) -> Dict[str, Any]:
        num_users = 1
        for p in state["user_input"].split():
            if p.isdigit():
                num_users = int(p)
                break
        graph = self.api_manager.simulate_load_test_fn(num_users=num_users)
        state["response"] = getattr(graph, "textContent", "")
        return state

    def execute_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.api_manager.execute_workflow_fn(state.get("user_input", ""))
        state["response"] = getattr(graph, "textContent", "")
        return state

    def unknown_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["response"] = "Sorry, I didn't understand that."
        return state

    def build_graph(self) -> StateGraph:
        builder = StateGraph()
        builder.add_node("route_intent", self.route_intent)
        builder.set_entry_point("route_intent")

        builder.add_node("parse_openapi", self.parse_openapi)
        builder.add_node("generate_sequence", self.generate_sequence)
        builder.add_node("generate_payloads", self.generate_payloads)
        builder.add_node("answer_openapi", self.answer_openapi)
        builder.add_node("simulate_load_test", self.simulate_load_test)
        builder.add_node("execute_workflow", self.execute_workflow)
        builder.add_node("unknown_intent", self.unknown_intent)

        builder.add_conditional_edges(
            "route_intent",
            lambda state: state.get("next_step", "unknown_intent"),
            {
                "parse_openapi": "parse_openapi",
                "generate_sequence": "generate_sequence",
                "generate_payloads": "generate_payloads",
                "answer_openapi": "answer_openapi",
                "simulate_load_test": "simulate_load_test",
                "execute_workflow": "execute_workflow",
                "unknown_intent": "unknown_intent",
            },
        )

        builder.set_finish_point("*")
        return builder.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from utils.openapi_utils import load_spec_text_from_file

    spec_file = os.getenv("OPENAPI_SPEC", "petstore.yaml")
    spec = load_spec_text_from_file(spec_file)

    router_llm = AzureChatOpenAI(deployment_name="gpt-35-router", temperature=0)
    worker_llm = AzureChatOpenAI(deployment_name="gpt-35-worker", temperature=0)

    router = LangGraphOpenApiRouter(spec, router_llm, worker_llm)
    graph = router.build_graph()

    while True:
        user_in = input("You: ")
        out = graph.invoke({"user_input": user_in})
        print("Assistant:", out.get("response", ""))
