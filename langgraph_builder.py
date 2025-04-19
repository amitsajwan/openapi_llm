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

        # Build and compile graph with MemorySaver checkpointing
        builder = StateGraph(GraphState)
        self._build_graph(builder)
        self.runner = builder.compile(checkpointer=MemorySaver())  # :contentReference[oaicite:2]{index=2}

    def _build_graph(self, builder: StateGraph):
        for node in self.graph_def["nodes"]:
            builder.add_node(
                node["operationId"],
                RunnableLambda(self._make_node_runner(node))
            )
        for edge in self.graph_def["edges"]:
            builder.add_edge(edge["from_node"], edge["to_node"])
        # Entry point
        entry = self.graph_def["nodes"][0]["operationId"]
        builder.set_entry_point(entry)

    def _make_node_runner(self, node: Dict[str, Any]):
        async def run_node(state: GraphState) -> GraphState:
            # 1) Prepare initial payload
            initial = {
                "operationId": node["operationId"],
                "method": node["method"],
                "path": node["path"],
                "payload": node.get("payload", {})
            }

            # 2) Pause for human to confirm/update
            confirmed: Dict[str, Any] = interrupt(initial)  # single-arg usage :contentReference[oaicite:3]{index=3}

            # 3) Execute API with confirmed payload
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed.get("payload", initial["payload"])
            )

            # 4) Extract IDs, merge into state
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
        if config is None or "thread_id" not in config:
            raise ValueError("config must include 'thread_id' for checkpointing")

        state = initial_state or GraphState()
        async for step in self.runner.astream(state.dict(), config):  # :contentReference[oaicite:4]{index=4}
            # Handle interrupts (human-in-the-loop)
            if isinstance(step, dict) and step.get("__type__") == "interrupt":
                intr = step["__interrupt__"][0]
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": intr.ns[0],
                    "operationId": intr.ns[0].split(":")[0],
                    "prompt": intr.value if isinstance(intr.value, str) else intr.value.get("prompt"),
                    "payload": intr.value.get("payload") if isinstance(intr.value, dict) else None
                })
                continue

            # Send new API results
            new_ops = set(step["operation_results"]) - set(state.operation_results)
            for op in new_ops:
                await self.websocket_callback("api_response", {
                    "operationId": op,
                    "result": step["operation_results"][op]
                })

            state = GraphState(**step)
            yield state

    async def submit_interrupt_response(self, key: str, resume_value: Any, config: Dict[str, Any]):
        # Resume graph at paused interrupt
        await self.runner.submit(Command(resume=resume_value), config)
