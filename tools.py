import json
from types import SimpleNamespace
from typing import Any, Dict, Optional

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

class ApiGraphManager:
    def __init__(self, llm: AzureChatOpenAI, openapi_yaml: str):
        self.llm = llm
        self.openapi_yaml = openapi_yaml

    def parse_spec_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses OpenAPI YAML text into a JSON-like dict.
        Caches result into state["openapi_schema"].
        """
        prompt = (
            "Convert the following OpenAPI YAML into a JSON dictionary representation:\n" +
            self.openapi_yaml
        )
        resp = self.llm([HumanMessage(content=prompt)])
        schema = json.loads(resp.content)
        state["openapi_schema"] = schema
        return state

    def generate_api_execution_graph_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an execution sequence (plan) of endpoints based on spec and optional user guidance.
        Stores result in state["execution_graph"].
        """
        guidance = state.get("user_input", "")
        prompt = (
            "Given this OpenAPI spec, generate an ordered list of API calls (method + path) as JSON: { \"sequence\": [...] }."
            f"\nSpec: {self.openapi_yaml}\nGuidance: {guidance}"
        )
        resp = self.llm([HumanMessage(content=prompt)])
        data = json.loads(resp.content)
        state["execution_graph"] = data.get("sequence")
        return state

    def generate_payload_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses LLM to generate realistic example payloads for each endpoint in spec.
        Stores mapping endpoint->payload in state["payloads"].
        """
        guidance = state.get("user_input", "")
        prompt = (
            "For each endpoint in this OpenAPI spec, generate a sample JSON payload."
            f"\nSpec: {self.openapi_yaml}\nContext: {guidance}"
        )
        resp = self.llm([HumanMessage(content=prompt)])
        payloads = json.loads(resp.content)
        state["payloads"] = payloads
        return state

    def openapi_help_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answers general questions about the API spec. Stores answer text in state["response"].
        """
        question = state.get("user_input", "")
        prompt = (
            "Answer the following question about the given OpenAPI spec:\n"
            f"Spec: {self.openapi_yaml}\nQuestion: {question}"
        )
        resp = self.llm([HumanMessage(content=prompt)])
        state["response"] = resp.content
        return state

    def simulate_load_test_fn(self, state: Dict[str, Any], num_users: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulates a load test with `num_users`. Stores summary in state["response"].
        """
        users = num_users or state.get("num_users", 1)
        prompt = f"Simulate a load test of {users} users on this API spec: {self.openapi_yaml}. Provide summary."  
        resp = self.llm([HumanMessage(content=prompt)])
        state["response"] = resp.content
        return state

    def execute_workflow_fn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produces a dry-run summary of executing the planned sequence. 
        Returns summary in state["response"].
        Actual HTTP calls occur outside.
        """
        seq = state.get("execution_graph", [])
        prompt = (
            "Provide a dry-run summary of executing these API calls in order:\n" + json.dumps(seq)
        )
        resp = self.llm([HumanMessage(content=prompt)])
        state["response"] = resp.content
        return state
