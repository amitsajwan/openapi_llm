from langgraph.graph import StateGraph, END, RunnableLambda
from langgraph.checkpoint import interrupt
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

# -------------------------------
# Schema Definitions
# -------------------------------
class Node(BaseModel):
    operationId: str
    method: str
    path: str
    payload: Optional[dict] = Field(default_factory=dict)

class Edge(BaseModel):
    from_node: str
    to_node: str
    can_parallel: bool
    reason: Optional[str] = None
    requires_human_validation: bool = False

class GraphOutput(BaseModel):
    nodes: list[Node]
    edges: list[Edge]

@dataclass
class GraphState:
    operation_results: Dict[str, Any]
    extracted_ids: Dict[str, str]

# -------------------------------
# GraphBuilder (LangGraph Generator)
# -------------------------------
class GraphBuilder:
    def __init__(self, graph_def: GraphOutput, api_executor, websocket_callback: Callable[[str, Any], None]):
        self.graph_def = graph_def
        self.api_executor = api_executor
        self.websocket_callback = websocket_callback
        self.state_graph = StateGraph(GraphState)

    def build(self):
        for node in self.graph_def.nodes:
            self.state_graph.add_node(
                node.operationId,
                RunnableLambda(self._make_node_runner(node))
            )

        for edge in self.graph_def.edges:
            if edge.requires_human_validation:
                self.state_graph.add_conditional_edges(
                    edge.from_node,
                    lambda state: "human_decision",
                    {
                        "human_decision": edge.to_node
                    }
                )
            else:
                self.state_graph.add_edge(edge.from_node, edge.to_node)

        terminal_nodes = self._find_terminal_nodes()
        for node_id in terminal_nodes:
            self.state_graph.add_edge(node_id, END)

        return self.state_graph

    def _make_node_runner(self, node: Node):
        async def run_node(state: GraphState) -> GraphState:
            payload = node.payload or {}

            # Ask user to confirm/edit payload via WebSocket
            confirmed_payload = await self._request_payload_confirmation(node, payload)

            try:
                result = await self.api_executor.execute_api(
                    method=node.method,
                    endpoint=node.path,
                    payload=confirmed_payload
                )
                self.websocket_callback("api_result", {
                    "operationId": node.operationId,
                    "message": f"{node.operationId} executed successfully.",
                    "response": result
                })
                extracted = self._extract_ids(result)
                return GraphState(
                    operation_results={**state.operation_results, node.operationId: result},
                    extracted_ids={**state.extracted_ids, **extracted}
                )
            except Exception as e:
                self.websocket_callback("api_result", {
                    "operationId": node.operationId,
                    "message": f"{node.operationId} failed: {str(e)}",
                    "response": {}
                })
                return state
        return run_node

    async def _request_payload_confirmation(self, node: Node, payload: dict) -> dict:
        updated_payload = await interrupt(
            f"Confirm payload for {node.operationId}",
            payload
        )
        return updated_payload

    def _extract_ids(self, result: dict) -> Dict[str, str]:
        ids = {}
        if isinstance(result, dict):
            for key, val in result.items():
                if key.endswith("id") and isinstance(val, (str, int)):
                    ids[key] = str(val)
        return ids

    def _find_terminal_nodes(self):
        from_nodes = {edge.from_node for edge in self.graph_def.edges}
        to_nodes = {edge.to_node for edge in self.graph_def.edges}
        return [n.operationId for n in self.graph_def.nodes if n.operationId not in from_nodes - to_nodes]


# -------------------------------
# LangGraphWorkflow (Runner Wrapper)
# -------------------------------
class LangGraphWorkflow:
    def __init__(self, graph_def: GraphOutput, api_executor, websocket_callback: Callable[[str, Any], None]):
        self.builder = GraphBuilder(graph_def, api_executor, websocket_callback)
        self.state_graph = self.builder.build().compile()

    async def run(self, initial_state: Optional[GraphState] = None):
        state = initial_state or GraphState(operation_results={}, extracted_ids={})
        async for step in self.state_graph.astream(state):
            yield step
