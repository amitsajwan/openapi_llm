from typing import Any, Callable, Awaitable, Dict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableLambda
from api_executor import APIExecutor
import asyncio

class GraphState(BaseModel):
    operations: List[Dict[str, Any]] = []
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
        self._resume_queue: asyncio.Queue = asyncio.Queue()

        builder = StateGraph(GraphState)
        self._build_graph(builder)
        self.runner = builder.compile(checkpointer=MemorySaver())

    def _build_graph(self, builder: StateGraph):
        for node in self.graph_def["nodes"]:
            builder.add_node(
                node["operationId"],
                RunnableLambda(self._make_runner(node))
            )
        for edge in self.graph_def["edges"]:
            builder.add_edge(edge["from_node"], edge["to_node"])
        entry = self.graph_def["nodes"][0]["operationId"]
        builder.set_entry_point(entry)

    def _make_runner(self, node: Dict[str, Any]):
        async def run_node(state: GraphState) -> Dict[str, Any]:
            initial = {
                "operationId": node["operationId"],
                "method": node["method"],
                "path": node["path"],
                "payload": node.get("payload", {}),
            }

            confirmed: Dict[str, Any] = interrupt(initial)

            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed.get("payload", initial["payload"])
            )

            extracted = self.api_executor.extract_ids(result)

            return {
                "operations": [{"operation_id": node["operationId"], "result": result}],
                "extracted_ids": {**state.extracted_ids, **extracted}
            }
        return run_node

    async def astream(
        self,
        initial_state: Optional[GraphState] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Streams execution, handles interrupts and resumes in‑stream.
        """
        state_dict = (initial_state or GraphState()).dict()
        cfg = config or {}
        if "thread_id" not in cfg:
            raise ValueError("Missing thread_id in config for checkpointing")

        gen = self.runner.astream(state_dict, cfg)
        while True:
            try:
                step = await gen.__anext__()
            except StopAsyncIteration:
                break

            # 1) Handle interrupt pause
            if "__interrupt__" in step:
                intr = step["__interrupt__"][0]
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": intr.ns[0],
                    "prompt": intr.value.get("question", "Confirm payload"),
                    "payload": intr.value
                })
                resume_val = await self._resume_q.get()
                # Resume in‑stream using Command(resume)
                gen = self.runner.astream(Command(resume=resume_val), cfg)  # continue streaming :contentReference[oaicite:3]{index=3}
                continue

            # 2) Emit new API results
            for op in step.get("operations", []):
                await self.websocket_callback("api_response", {
                    "operationId": op["operation_id"],
                    "result": op["result"]
                })

            state_dict = step
            yield GraphState(**step)


    async def submit_resume(self, resume_value: Any):
        await self._resume_queue.put(resume_value)
