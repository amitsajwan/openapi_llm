# langgraph_builder.py
from typing import Any, Dict, Callable
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
    def __init__(self, graph_def: Dict[str, Any], api_executor: APIExecutor):
        self.graph_def = graph_def
        self.api_executor = api_executor

    def build_runner(self):
        builder = StateGraph(GraphState)

        # 1) Add one node per operation
        for node in self.graph_def["nodes"]:
            builder.add_node(
                node["operationId"],
                RunnableLambda(self._make_runner(node))
            )

        # 2) Wire edges (dependencies)
        for edge in self.graph_def["edges"]:
            builder.add_edge(edge["from_node"], edge["to_node"])

        # 3) Set entry point
        entry = self.graph_def["nodes"][0]["operationId"]
        builder.set_entry_point(entry)

        # 4) Compile with in‑memory checkpointing for interrupts
        runner = builder.compile(checkpointer=MemorySaver())
        return runner

    def _make_runner(self, node: Dict[str, Any]) -> Callable[[GraphState], GraphState]:
        async def run_node(state: GraphState) -> GraphState:
            # Prepare the initial payload dict
            initial = {
                "operationId": node["operationId"],
                "method": node["method"],
                "path": node["path"],
                "payload": node.get("payload", {})
            }

            # Pause here and ask the human to confirm/edit
            confirmed: Dict[str, Any] = interrupt(initial)

            # Execute the API with the confirmed payload
            result = await self.api_executor.execute_api(
                method=node["method"],
                endpoint=node["path"],
                payload=confirmed.get("payload", initial["payload"])
            )

            # Extract any IDs for placeholders in downstream calls
            extracted = self.api_executor.extract_ids(result)

            # Return updated state
            return GraphState(
                operation_results={**state.operation_results, node["operationId"]: result},
                extracted_ids={**state.extracted_ids, **extracted}
            )

        return run_node
