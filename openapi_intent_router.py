from typing import List, Dict, Any
import json
import asyncio

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langgraph.graph import State


class RouterState(State):
    query: str
    intent: str = ""
    history: List[Dict[str, str]] = []


class OpenAPIIntentRouterAgent:
    def __init__(
        self,
        llm: AzureChatOpenAI,
        openapi_spec: Dict[str, Any],
    ):
        self.llm = llm
        self.openapi_spec = openapi_spec
        self.graph = self._build_agentic_graph()

    def _build_agentic_graph(self) -> StateGraph:
        """Construct an agentic StateGraph with modular intent handlers."""
        graph = StateGraph(RouterState)

        # 1. Triage node: classify intent
        async def classify_intent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """
You are an AI assistant that classifies user queries into intents.
Based on conversation history, choose one of:
- general_inquiry
- openapi_help
- generate_payload
- generate_sequence
- create_workflow
- execute_workflow

Return JSON: {\"intent\": "intent_name"}

History: {history}
Query: {query}
"""
            ).format(query=state.query, history=state.history)

            response = await self.llm.ainvoke(prompt)
            intent = json.loads(response).get("intent", "general_inquiry")
            return RouterState(query=state.query, intent=intent, history=state.history)

        # 2. Define specialized agents
        async def general_inquiry_agent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """Answer general questions: {query}"""
            ).format(query=state.query)
            bot_reply = await self.llm.ainvoke(prompt)
            return self._append_history(state, bot_reply)

        async def openapi_help_agent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """Extract API details from spec and answer: {query}\nSpec: {spec}"""
            ).format(query=state.query, spec=self.openapi_spec)
            bot_reply = await self.llm.ainvoke(prompt)
            return self._append_history(state, bot_reply)

        async def generate_payload_agent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """Generate a valid JSON payload for endpoint '{query}' using spec: {spec}"""
            ).format(query=state.query, spec=self.openapi_spec)
            bot_reply = await self.llm.ainvoke(prompt)
            return self._append_history(state, bot_reply)

        async def generate_sequence_agent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """Determine API call sequence using spec: {spec}"""
            ).format(spec=self.openapi_spec)
            bot_reply = await self.llm.ainvoke(prompt)
            return self._append_history(state, bot_reply)

        async def create_workflow_agent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """Construct a LangGraph workflow from spec: {spec}"""
            ).format(spec=self.openapi_spec)
            bot_reply = await self.llm.ainvoke(prompt)
            return self._append_history(state, bot_reply)

        async def execute_workflow_agent(state: RouterState) -> RouterState:
            prompt = PromptTemplate.from_template(
                """Execute the predefined workflow as instructed."""
            ).format(spec=self.openapi_spec)
            bot_reply = await self.llm.ainvoke(prompt)
            return self._append_history(state, bot_reply)

        # 3. Routing function
        def route_by_intent(state: RouterState) -> str:
            return state.intent or "general_inquiry"

        # Register nodes
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("route_by_intent", route_by_intent)
        graph.add_node("general_inquiry", general_inquiry_agent)
        graph.add_node("openapi_help", openapi_help_agent)
        graph.add_node("generate_payload", generate_payload_agent)
        graph.add_node("generate_sequence", generate_sequence_agent)
        graph.add_node("create_workflow", create_workflow_agent)
        graph.add_node("execute_workflow", execute_workflow_agent)

        # Define flow
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "route_by_intent")
        graph.add_conditional_edges(
            "route_by_intent",
            condition=route_by_intent,
            path_map={
                "general_inquiry": "general_inquiry",
                "openapi_help": "openapi_help",
                "generate_payload": "generate_payload",
                "generate_sequence": "generate_sequence",
                "create_workflow": "create_workflow",
                "execute_workflow": "execute_workflow",
            },
        )

        return graph.compile()

    def _append_history(self, state: RouterState, bot_reply: str) -> RouterState:
        new_hist = state.history + [{"user": state.query, "bot": bot_reply}]
        return RouterState(query=state.query, intent=state.intent, history=new_hist)

    async def handle_user_input(
        self, user_input: str, history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        if history is None:
            history = []
        state = RouterState(query=user_input, history=history)
        updated_state = await self._invoke_graph(state)
        latest = updated_state.history[-1]["bot"] if updated_state.history else ""
        return {"response": latest, "history": updated_state.history}

    async def _invoke_graph(self, state: RouterState) -> RouterState:
        # Support async invoke if available
        if callable(getattr(self.graph, "ainvoke", None)):
            return await self.graph.ainvoke(state)
        return self.graph.invoke(state)

# Usage example (async context):
# llm = AzureChatOpenAI(deployment_name="gpt4o", temperature=0)\#
# router = OpenAPIIntentRouterAgent(llm, openapi_spec_dict)
# result = await router.handle_user_input("List all endpoints", [])
# print(result)
