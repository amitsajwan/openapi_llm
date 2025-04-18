
from typing import Any, Callable, Awaitable, Dict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runner import StreamEvent, StreamDelta, SubmitInterruptRequest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.schema import GraphContext
from api_executor import APIExecutor

class GraphState(BaseModel):
    operation_results: Dict[str, Any] = {}
    extracted_ids: Dict[str, Any] = {}

class LangGraphWorkflow:
    def __init__(
        self,
        graph_def: Dict[str, Any],
        api_executor: APIExecutor,
        websocket_callback: Callable[[str, Dict[str, Any]], Awaitable[None]],
    ):
        self.graph_def = graph_def
        self.api_executor = api_executor
        self.websocket_callback = websocket_callback
        self.state_graph = self._build_graph()
        self.runner = self.state_graph.compile(checkpointer=MemorySaver())

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(GraphState)
        # Add nodes
        for node in self.graph_def['nodes']:
            builder.add_node(
                node['operationId'],
                RunnableLambda(self._make_node_runner(node))
            )
        # Add edges
        for edge in self.graph_def['edges']:
            builder.add_edge(edge['from_node'], edge['to_node'])
        # Entry point
        entry_node = self.graph_def['nodes'][0]['operationId']
        builder.set_entry_point(entry_node)
        return builder

    def _make_node_runner(self, node: Dict[str, Any]) -> Callable[[GraphState], Awaitable[GraphState]]:
        async def run_node(state: GraphState) -> GraphState:
            payload = node.get("payload", {})
            # Human-in-loop payload confirmation
            confirmed = await GraphContext.interrupt(
                f"Confirm payload for {node['operationId']}",
                {
                    "operationId": node["operationId"],
                    "method": node["method"],
                    "path": node["path"],
                    "payload": payload,
                }
            )
            actual_payload = confirmed.get("payload", payload)
            # Execute API
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=actual_payload
            )
            # Extract IDs
            extracted = self.api_executor.extract_ids(result)
            # Return new state
            return GraphState(
                operation_results={**state.operation_results, node["operationId"]: result},
                extracted_ids={**state.extracted_ids, **extracted}
            )
        return run_node

    async def astream(self, initial_state: Optional[GraphState] = None):
        state = initial_state or GraphState()
        async for step in self.runner.astream(state):
            # Handle interrupt events
            if isinstance(step, dict) and step.get("__type__") == "interrupt":
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": step["key"],
                    "prompt": step["name"],
                    "operationId": step["content"]["operationId"],
                    "payload": step["content"]["payload"]
                })
                continue
            # Send new API results
            new_ops = set(step.operation_results.keys()) - set(state.operation_results.keys())
            for op_id in new_ops:
                await self.websocket_callback("api_response", {
                    "operationId": op_id,
                    "result": step.operation_results[op_id]
                })
            state = step
            yield step

    async def submit_interrupt_response(self, key: str, value: Dict[str, Any]):
        await self.runner.submit_interrupt(key, value)
