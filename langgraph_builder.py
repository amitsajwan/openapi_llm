from typing import Any, Callable, Awaitable, Dict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runner import StreamEvent, StreamDelta, SubmitInterruptRequest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.schema import GraphContext
from api_executor import APIExecutor
from pydantic import BaseModel
from fastapi import WebSocket
import asyncio


# ----------------------
# 1. Define Graph State
# ----------------------

class GraphState(BaseModel):
    operation_results: Dict[str, Any] = {}
    extracted_ids: Dict[str, Any] = {}


# ----------------------------
# 2. Define LangGraph Workflow
# ----------------------------

class LangGraphWorkflow:
    def __init__(
        self,
        graph_def: Dict,
        api_executor: APIExecutor,
        websocket_callback: Callable[[str, Dict[str, Any]], Awaitable[None]],
    ):
        self.graph_def = graph_def
        self.api_executor = api_executor
        self.websocket_callback = websocket_callback

        self.state_graph = self._build_graph()
        self.runner = self.state_graph.compile()

    # ------------------------
    # 3. Build LangGraph nodes
    # ------------------------
    def _build_graph(self) -> StateGraph:
        builder = StateGraph(GraphState)
        nodes = {node['operationId']: node for node in self.graph_def['nodes']}

        for op_id, node in nodes.items():
            builder.add_node(op_id, RunnableLambda(self.make_node_runner(node)))

        for edge in self.graph_def['edges']:
            if edge['can_parallel']:
                builder.add_edge(edge['from_node'], edge['to_node'])
            else:
                builder.add_edge(edge['from_node'], edge['to_node'])

        entry_node = self.graph_def['nodes'][0]['operationId']
        builder.set_entry_point(entry_node)
        return builder

    # --------------------
    # 4. Node Runner Logic
    # --------------------
    def make_node_runner(self, node: Dict[str, Any]) -> Callable[[GraphState], Awaitable[GraphState]]:
        async def run_node(state: GraphState) -> GraphState:
            payload = node.get("payload", {})

            # Human-in-the-loop confirmation using LangGraph's interrupt
            confirmed_payload = await GraphContext.interrupt(
                f"Confirm payload for {node['operationId']}",
                {
                    "operationId": node["operationId"],
                    "method": node["method"],
                    "path": node["path"],
                    "payload": payload,
                }
            )

            # Execute API
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed_payload.get("payload", payload)
            )

            # Extract IDs from result for chaining
            extracted = self.api_executor.extract_ids(result)

            return GraphState(
                operation_results={**state.operation_results, node["operationId"]: result},
                extracted_ids={**state.extracted_ids, **extracted}
            )

        return run_node

    # ---------------------------------
    # 5. Async stream with WebSocket IO
    # ---------------------------------
    async def astream(self, state: GraphState):
        async for step in self.runner.astream(state):
            if isinstance(step, dict) and step.get("__type__") == "interrupt":
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": step["key"],
                    "prompt": step["name"],
                    "operationId": step["content"]["operationId"],
                    "payload": step["content"]["payload"]
                })
                continue

            # Handle API result (only new ones)
            delta_keys = step.operation_results.keys() - state.operation_results.keys()
            for op_id in delta_keys:
                await self.websocket_callback("api_result", {
                    "operationId": op_id,
                    "result": step.operation_results[op_id]
                })

            state = step
            yield step

    # ---------------------------------------
    # 6. Submit human response for interrupt
    # ---------------------------------------
    async def submit_interrupt_response(self, key: str, value: Dict[str, Any]):
        await self.runner.submit_interrupt(key, value)
