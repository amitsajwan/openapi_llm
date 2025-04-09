import asyncio
import httpx
import re
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    first_run: bool = True
    confirmed: Dict[str, bool] = Field(default_factory=dict)
    payloads: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    base_url: str = "http://localhost:8000"
    headers: Dict[str, str] = Field(default_factory=lambda: {"Content-Type": "application/json"})


class LanggraphWorkflow:
    def __init__(self, graph_json: Dict[str, Any], base_url: str = "http://localhost:8000"):
        self.graph_json = graph_json
        self.node_configs = {node["id"]: node for node in graph_json["nodes"]}
        self.base_url = base_url
        self.entry_nodes = self._find_entry_nodes()
        self.terminal_nodes = self._find_terminal_nodes()
        self.graph = self._build_graph()

    def _find_entry_nodes(self) -> List[str]:
        sources = {edge["source"] for edge in self.graph_json["edges"]}
        targets = {edge["target"] for edge in self.graph_json["edges"]}
        return list(sources - targets) or [list(self.node_configs.keys())[0]]

    def _find_terminal_nodes(self) -> List[str]:
        sources = {edge["source"] for edge in self.graph_json["edges"]}
        targets = {edge["target"] for edge in self.graph_json["edges"]}
        return list(targets - sources)

    def _extract_path_variables(self, path: str) -> List[str]:
        return re.findall(r"{(.*?)}", path)

    def _resolve_path_variables(self, path: str, variables: Dict[str, Any]) -> str:
        for var in self._extract_path_variables(path):
            if var not in variables:
                raise ValueError(f"Missing variable '{var}' for path: {path}")
            path = path.replace(f"{{{var}}}", str(variables[var]))
        return path

    async def _get_payload_for_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        # Replace with your actual payload generator
        return {"sample": f"generated_for_{endpoint_id}"}

    def _build_node_fn(self, config: Dict[str, Any]):
        async def _node_fn(state: GraphState) -> GraphState:
            method = config["method"].upper()
            endpoint = config["endpoint"]
            node_id = config["id"]

            try:
                url = self._resolve_path_variables(endpoint, state.variables)
            except ValueError as e:
                print(f"[{node_id}] Error: {e}")
                return state

            full_url = state.base_url + url
            payload = {}

            if method in ["POST", "PUT"]:
                if state.first_run and not state.confirmed.get(node_id):
                    payload = await self._get_payload_for_endpoint(node_id)
                    return state.copy(update={
                        "action": "confirm_payload",
                        "target_node": node_id,
                        "payload": payload
                    })
                payload = state.payloads.get(node_id, await self._get_payload_for_endpoint(node_id))

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method,
                    full_url,
                    headers=state.headers,
                    json=payload if method in ["POST", "PUT"] else None
                )

            try:
                data = response.json()
            except Exception:
                data = {"error": response.text}

            new_vars = {**state.variables}
            if isinstance(data, dict) and "id" in data:
                new_vars["id"] = data["id"]

            return state.copy(update={
                "results": {**state.results, node_id: data},
                "variables": new_vars
            })

        return RunnableLambda(_node_fn)

    def _build_graph(self):
        sg = StateGraph(state_schema=GraphState)

        # Add all nodes
        for node_id, config in self.node_configs.items():
            sg.add_node(node_id, self._build_node_fn(config))

        # Add edges
        for edge in self.graph_json["edges"]:
            sg.add_edge(edge["source"], edge["target"])

        # Add END to terminal nodes
        for terminal in self.terminal_nodes:
            sg.add_edge(terminal, END)

        # Entry point
        sg.set_entry_point(self.entry_nodes[0])
        return sg.compile()

    async def run(self):
        state = GraphState(base_url=self.base_url)

        while True:
            output = await self.graph.invoke(state.model_dump())

            # Handle payload confirmation
            if isinstance(output, dict) and output.get("action") == "confirm_payload":
                node_id = output["target_node"]
                payload = output["payload"]
                print(f"\nPayload for node '{node_id}': {payload}")
                modified = await self.user_confirm_payload(node_id, payload)
                state.payloads[node_id] = modified
                state.confirmed[node_id] = True
                state.first_run = False
                continue

            print("\n--- Final Results ---")
            for k, v in output["results"].items():
                print(f"{k}: {v}")
            return output

    async def user_confirm_payload(self, node_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Please confirm or edit payload for node '{node_id}':")
        print(payload)
        # Simulate user confirmation - replace this with UI/CLI logic
        await asyncio.sleep(0.5)
        payload["sample"] = f"user_modified_{node_id}"
        return payload
