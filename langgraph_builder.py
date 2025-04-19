from typing import Any, Callable, Awaitable, Dict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableLambda
from api_executor import APIExecutor

# ----------------------
# 1. Define Graph State
# ----------------------
class GraphState(BaseModel):
    # tracks completed API results
    operation_results: Dict[str, Any] = {}
    # carries extracted IDs for path placeholders
    extracted_ids: Dict[str, Any] = {}  

# ----------------------------
# 2. Define LangGraph Workflow
# ----------------------------
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

        # Build and compile the StateGraph with in‑memory checkpointing
        builder = StateGraph(GraphState)  :contentReference[oaicite:1]{index=1}
        self._build_graph(builder)
        # Attach a MemorySaver for persistence (enables interrupts) 
        self.runner = builder.compile(checkpointer=MemorySaver())  :contentReference[oaicite:2]{index=2}

    # ------------------------
    # 3. Build LangGraph nodes
    # ------------------------
    def _build_graph(self, builder: StateGraph):
        # Add each operation as a node
        for node in self.graph_def["nodes"]:
            builder.add_node(
                node["operationId"],
                RunnableLambda(self._make_node_runner(node))
            )  :contentReference[oaicite:3]{index=3}

        # Wire up dependencies
        for edge in self.graph_def["edges"]:
            builder.add_edge(edge["from_node"], edge["to_node"])

        # Set the first operation as entry point
        entry = self.graph_def["nodes"][0]["operationId"]
        builder.set_entry_point(entry)

    # --------------------
    # 4. Node Runner Logic
    # --------------------
    def _make_node_runner(self, node: Dict[str, Any]) -> Callable[[GraphState], GraphState]:
        async def run_node(state: GraphState) -> GraphState:
            # 4.1 Prepare payload (may contain placeholders)
            payload = node.get("payload", {})

            # 4.2 Pause for human confirmation/edit of the payload 
            # (interrupt returns the user's reply) :contentReference[oaicite:4]{index=4}
            confirmed = interrupt(
                f"Confirm payload for {node['operationId']}",
                {
                    "operationId": node["operationId"],
                    "method": node["method"],
                    "path": node["path"],
                    "payload": payload
                }
            )

            # 4.3 Execute the API (after payload confirmation)
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed.get("payload", payload)
            )

            # 4.4 Extract IDs for future placeholders
            extracted = self.api_executor.extract_ids(result)

            # 4.5 Return updated GraphState
            return GraphState(
                operation_results={**state.operation_results, node["operationId"]: result},
                extracted_ids={**state.extracted_ids, **extracted}
            )
        return run_node

    # ---------------------------------
    # 5. Async stream with WebSocket IO
    # ---------------------------------
    async def astream(
        self,
        initial_state: Optional[GraphState] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        # Ensure we have a state and config with thread_id for checkpointing
        state = initial_state or GraphState()
        cfg = config or {}
        if "thread_id" not in cfg:
            raise ValueError("`thread_id` required in config for checkpointing")

        # Stream through the graph; catches interrupts and completions
        async for step in self.runner.astream(state.dict(), cfg):  :contentReference[oaicite:5]{index=5}
            # 5.1 Handle human‑in‑the‑loop interrupt events
            if isinstance(step, dict) and step.get("__type__") == "interrupt":
                intr = step["__interrupt__"][0]
                await self.websocket_callback("payload_confirmation", {
                    "interrupt_key": intr.value,  # use intr.value or intr.ns[0]
                    "operationId": intr.ns[0].split(":")[0],
                    "prompt": intr.value if isinstance(intr.value, str) else intr.value.get("payload"),
                    "payload": intr.value.get("payload") if isinstance(intr.value, dict) else None
                })
                continue

            # 5.2 On API node completion, send results
            new_ops = set(step.get("operation_results", {})) - set(state.operation_results)
            for op_id in new_ops:
                await self.websocket_callback("api_response", {
                    "operationId": op_id,
                    "result": step["operation_results"][op_id]
                })

            # 5.3 Yield updated state for logging/testing
            state = GraphState(**step)  
            yield state

    # -------------------------------------------------
    # 6. Submit human response for interrupt to resume
    # -------------------------------------------------
    async def submit_interrupt_response(
        self, interrupt_key: str, resume_value: Any, config: Dict[str, Any]
    ):
        # Use LangGraph's Command primitive to resume from the interrupt :contentReference[oaicite:6]{index=6}
        await self.runner.submit( 
            Command(resume=resume_value),
            config
        )
