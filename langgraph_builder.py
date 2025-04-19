from typing import Any, Callable, Awaitable, Dict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableLambda
from api_executor import APIExecutor
import asyncio

class GraphState(BaseModel):
    operation_results: Dict[str, Any] = {}
    extracted_ids: Dict[str, Any] = {}

class LangGraphBuilder:
    def __init__(
        self,
        graph_def: Dict[str, Any],
        api_executor: APIExecutor,
        websocket_callback: Callable[[str, Dict[str, Any]], Awaitable[None]],
    ):
        self.graph_def = graph_def
        self.api_executor = api_executor
        self.websocket_callback = websocket_callback
        # Queue to receive the user’s resume payload
        self._resume_queue: asyncio.Queue = asyncio.Queue()

        # Build & compile
        builder = StateGraph(GraphState)
        self._build_graph(builder)
        self.runner = builder.compile(checkpointer=MemorySaver())

    def _build_graph(self, builder: StateGraph):
        # add nodes
        for node in self.graph_def["nodes"]:
            builder.add_node(
                node["operationId"],
                RunnableLambda(self._make_runner(node))
            )
        # add edges
        for edge in self.graph_def["edges"]:
            builder.add_edge(edge["from_node"], edge["to_node"])
        # entry
        entry = self.graph_def["nodes"][0]["operationId"]
        builder.set_entry_point(entry)

    def _make_runner(self, node: Dict[str, Any]):
        async def run_node(state: GraphState) -> GraphState:
            # initial payload dict
            initial = {
                "operationId": node["operationId"],
                "method": node["method"],
                "path": node["path"],
                "payload": node.get("payload", {}),
            }

            # pause for human confirmation
            confirmed: Dict[str, Any] = interrupt(initial)

            # execute API
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed.get("payload", initial["payload"])
            )

            # extract IDs
            extracted = self.api_executor.extract_ids(result)

            return GraphState(
                operation_results={**state.operation_results, node["operationId"]: result},
                extracted_ids={**state.extracted_ids, **extracted}
            )
        return run_node

    async def astream(
        self,
        initial_state: Optional[GraphState] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        state = initial_state or GraphState()
        cfg = config or {}
        if "thread_id" not in cfg:
            raise ValueError("config must include 'thread_id' for checkpointing")

        async for step in self.runner.astream(state.dict(), cfg):
            # 1) Interrupt event
            if "__interrupt__" in step:
                intr = step["__interrupt__"][0]
                # send prompt + payload
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": intr.ns[0],
                    "prompt": intr.value.get("question", "Confirm payload"),
                    "payload": intr.value.get("payload", {})
                })
                # wait for user to reply via queue
                resume_value = await self._resume_queue.get()
                # resume graph
                await self.runner.submit(Command(resume=resume_value), cfg)
                continue

            # 2) API result events
            new_ops = set(step["operation_results"]) - set(state.operation_results)
            for op_id in new_ops:
                await self.websocket_callback("api_response", {
                    "operationId": op_id,
                    "result": step["operation_results"][op_id]
                })

            state = GraphState(**step)
            yield state

    async def submit_resume(self, resume_value: Any):
        """
        Called by main.py when user confirms payload.
        """
        await self._resume_queue.put(resume_value)
