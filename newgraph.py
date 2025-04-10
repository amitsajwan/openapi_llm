import json
import httpx
from typing import Any, Dict, List
from langgraph.graph import StateGraph
from langchain.chat_models import AzureChatOpenAI

class LanggraphWorkflow:
    def __init__(self, graph_def: Dict[str, Any], websocket, llm: AzureChatOpenAI):
        self.graph_def = graph_def
        self.websocket = websocket
        self.llm = llm
        self.state_graph = None

    def extract_terminal_nodes(self) -> List[str]:
        to_ids = {edge['to'] for edge in self.graph_def['edges']}
        from_ids = {edge['from'] for edge in self.graph_def['edges']}
        return list(to_ids - from_ids)

    def extract_entry_nodes(self) -> List[str]:
        to_ids = {edge['to'] for edge in self.graph_def['edges']}
        from_ids = {edge['from'] for edge in self.graph_def['edges']}
        return list(from_ids - to_ids)

    def build_graph(self):
        from langchain_core.runnables import RunnableLambda

        graph = StateGraph(state_schema=dict)
        node_map = {}

        for node in self.graph_def['nodes']:
            node_id = node['id']
            node_map[node_id] = RunnableLambda(self.make_node_runner(node))
            graph.add_node(node_id, node_map[node_id])

        for edge in self.graph_def['edges']:
            graph.add_edge(edge['from'], edge['to'])

        entry_nodes = self.extract_entry_nodes()
        if entry_nodes:
            graph.set_entry_point(entry_nodes[0])

        terminal_nodes = self.extract_terminal_nodes()
        for t in terminal_nodes:
            graph.set_finish_point(t)

        self.state_graph = graph.compile()

    def make_node_runner(self, node):
        async def run_node(state: Dict[str, Any]):
            method = node['type']
            operation_id = node['id']
            endpoint_template = node['endpoint']

            endpoint = endpoint_template
            for key, value in state.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        endpoint = endpoint.replace(f"{{{subkey}}}", str(subvalue))

            payload = await self.generate_payload(method, operation_id, endpoint)

            await self.websocket.send_json({
                "type": "executing_api",
                "sender": "bot",
                "message": f"Executing {method.upper()} {endpoint} [{operation_id}]",
                "method": method,
                "endpoint": endpoint,
                "operation_id": operation_id,
                "payload": payload,
            })

            async with httpx.AsyncClient() as client:
                try:
                    if method.lower() == "post":
                        resp = await client.post(endpoint, json=payload)
                    elif method.lower() == "put":
                        resp = await client.put(endpoint, json=payload)
                    elif method.lower() == "get":
                        resp = await client.get(endpoint, params=payload)
                    elif method.lower() == "delete":
                        resp = await client.delete(endpoint, params=payload)
                    else:
                        resp = None

                    try:
                        result = resp.json() if resp else {"error": "No response"}
                    except Exception:
                        result = {"error": "Failed to parse response"}
                except Exception as e:
                    result = {"error": str(e)}

            await self.websocket.send_json({
                "type": "api_response",
                "sender": "bot",
                "message": f"{method.upper()} {endpoint} executed. Response: {json.dumps(result)[:300]}...",
                "operation_id": operation_id,
                "response": result
            })

            return {**state, operation_id: result}

        return run_node

    async def generate_payload(self, method: str, operation_id: str, endpoint: str) -> Dict[str, Any]:
        prompt = f"""
Generate an example payload for the following API call.
Method: {method.upper()}
Operation ID: {operation_id}
Endpoint: {endpoint}
Return JSON only.
"""
        response = await self.llm.ainvoke(prompt)
        try:
            payload = json.loads(response.content.strip())
        except Exception:
            payload = {}
        return await self.confirm_payload(method, operation_id, endpoint, payload)

    async def confirm_payload(self, method: str, operation_id: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        current_payload = payload

        await self.websocket.send_json({
            "type": "confirm_payload",
            "sender": "bot",
            "method": method,
            "operation_id": operation_id,
            "endpoint": endpoint,
            "payload": current_payload,
            "message": (
                f"Suggested payload for {method.upper()} {endpoint} [{operation_id}].\n"
                "You can respond with changes like: change name to 'abc', remove 'id', or say 'looks good'."
            )
        })

        while True:
            user_msg = await self.websocket.receive_text()
            result = await llm_refiner(current_payload, user_msg, self.llm)

            if isinstance(result, dict):
                current_payload = result
                await self.websocket.send_json({
                    "type": "confirm_payload",
                    "sender": "bot",
                    "method": method,
                    "operation_id": operation_id,
                    "endpoint": endpoint,
                    "payload": current_payload,
                    "message": "Updated payload. Confirm or provide more instructions."
                })
            elif result == "__approved__":
                await self.websocket.send_json({
                    "type": "confirmation_done",
                    "sender": "bot",
                    "method": method,
                    "operation_id": operation_id,
                    "endpoint": endpoint,
                    "payload": current_payload,
                    "message": "Payload confirmed. Proceeding with API call."
                })
                return current_payload
            else:
                await self.websocket.send_json({
                    "type": "confirm_payload",
                    "sender": "bot",
                    "method": method,
                    "operation_id": operation_id,
                    "endpoint": endpoint,
                    "payload": current_payload,
                    "message": "Didn't understand. Please confirm or provide new instructions."
                })

    async def astream(self):
        async for state in self.state_graph.astream({}):
            await self.websocket.send_json({
                "type": "state_update",
                "sender": "bot",
                "message": f"Graph state updated with new data.",
                "state": state
            })


async def llm_refiner(payload: Dict[str, Any], user_instruction: str, llm: AzureChatOpenAI) -> Any:
    prompt = f"""
You're assisting in refining JSON payloads for API calls.

Current Payload:
{json.dumps(payload, indent=2)}

User Instruction: "{user_instruction}"

If approved, reply with __approved__.
If changes are needed, return only modified JSON.
"""
    response = await llm.ainvoke(prompt)
    content = response.content.strip()

    if content == "__approved__":
        return "__approved__"

    try:
        return json.loads(content)
    except Exception:
        return None
