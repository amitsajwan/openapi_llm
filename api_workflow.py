from langgraph.graph import StateGraph
from state import ApiExecutionState

class APIWorkflowManager:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers
        self.graph = StateGraph(ApiExecutionState)

    def build_workflow(self, api_sequence):
        for api in api_sequence:
            method, endpoint = api.split(" ", 1)

            async def node_fn(state, method=method, endpoint=endpoint):
                result = await state.executor.execute_api(method, endpoint)
                state.last_api = endpoint
                return state

            self.graph.add_node(endpoint, node_fn)

        for i in range(len(api_sequence) - 1):
            self.graph.add_edge(api_sequence[i], api_sequence[i + 1])

        return self.graph

    async def execute_workflow(self, api_sequence):
        self.build_workflow(api_sequence)
        state = ApiExecutionState()
        return await self.graph.run(state)
