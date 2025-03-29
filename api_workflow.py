import asyncio
from langgraph.graph import StateGraph
from api_executor import APIExecutor
from state import ApiExecutionState

class APIWorkflowManager:
    def __init__(self, openapi_data):
        self.openapi_data = openapi_data
        self.executor = APIExecutor(openapi_data)
        self.graph = StateGraph(ApiExecutionState)  # ✅ Use LangGraph for managing execution

    def build_workflow(self, sequence):
        """Builds the execution workflow dynamically from the sequence."""
        previous_step = None

        for step in sequence:
            method, endpoint = step["method"], step["endpoint"]

            async def node_fn(state, method=method, endpoint=endpoint):
                payload = step.get("payload", None)
                result = await self.executor.execute_api(method, endpoint, payload)
                state.last_api = endpoint
                return state

            self.graph.add_node(endpoint, node_fn)

            if previous_step:
                self.graph.add_edge(previous_step, endpoint)

            previous_step = endpoint

    async def execute_workflow(self, sequence):
        """Executes the API workflow using LangGraph."""
        self.build_workflow(sequence)
        state = ApiExecutionState()
        return await self.graph.run(state)
