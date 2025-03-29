from langgraph.graph import StateGraph
from state import ApiExecutionState

class APIWorkflowManager:
    def __init__(self):
        self.graph = StateGraph(ApiExecutionState)

    def build_workflow(self, sequence):
        previous = None
        for step in sequence:
            async def node_fn(state, step=step):
                result = await state.executor.execute_api("GET", step)
                state.last_api = step
                return state

            self.graph.add_node(step, node_fn)
            if previous:
                self.graph.add_edge(previous, step)
            previous = step
        return self.graph

    async def execute_workflow(self, sequence):
        self.build_workflow(sequence)
        state = ApiExecutionState()
        return await self.graph.run(state)
