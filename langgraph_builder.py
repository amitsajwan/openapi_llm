# langgraph_builder.py

from typing import Any, Callable, Awaitable, Dict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableLambda
from api_executor import APIExecutor


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

        # Build & compile graph with checkpointing enabled
        builder = StateGraph(GraphState)
        self._build_graph(builder)
        self.runner = builder.compile(checkpointer=MemorySaver())

    def _build_graph(self, builder: StateGraph):
        # 1) Add nodes
        for node in self.graph_def["nodes"]:
            builder.add_node(
                node["operationId"],
                RunnableLambda(self._make_runner(node))
            )

        # 2) Wire edges
        for edge in self.graph_def["edges"]:
            builder.add_edge(edge["from_node"], edge["to_node"])

        # 3) Set entry point
        entry = self.graph_def["nodes"][0]["operationId"]
        builder.set_entry_point(entry)

    def _make_runner(self, node: Dict[str, Any]):
        async def run_node(state: GraphState) -> GraphState:
            # Prepare payload
            initial = {
                "operationId": node["operationId"],
                "method": node["method"],
                "path": node["path"],
                "payload": node.get("payload", {}),
            }

            # 1) Human‑in‑loop interrupt
            confirmed: Dict[str, Any] = interrupt(initial)

            # 2) Execute API
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed.get("payload", initial["payload"])
            )

            # 3) Extract IDs for downstream
            extracted = self.api_executor.extract_ids(result)

            # 4) Return updated state
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
        """
        Streams through the graph, sends websocket messages on interrupts
        and on API completions—all via the injected websocket_callback.
        """
        state = initial_state or GraphState()
        cfg = config or {}
        if "thread_id" not in cfg:
            raise ValueError("`config` must include a unique 'thread_id' for checkpointing")

        async for step in self.runner.astream(state.dict(), cfg):
            # Handle human‑in‑the‑loop interrupt
            if "__interrupt__" in step:
                intr = step["__interrupt__"][0]
                # Send prompt + payload to client
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": intr.ns[0],
                    "prompt": intr.value.get("question", "Please confirm payload"),
                    "payload": intr.value.get("payload", {})
                })
                continue

            # On API node completion, send results
            new_ops = set(step["operation_results"]) - set(state.operation_results)
            for op_id in new_ops:
                await self.websocket_callback("api_response", {
                    "operationId": op_id,
                    "result": step["operation_results"][op_id]
                })

            # Update local state and yield
            state = GraphState(**step)
            yield state

    async def submit_interrupt_response(
        self,
        interrupt_key: str,
        resume_value: Any,
        config: Dict[str, Any]
    ):
        """
        Called by external WebSocket handler to resume
        from a human‑in‑the‑loop interrupt.
        """
        await self.runner.submit(Command(resume=resume_value), config)
