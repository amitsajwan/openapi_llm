import json
from typing import Any, Dict, Callable, List
from langgraph.graph import StateGraph

class LanggraphWorkflow:
    def __init__(self, graph_def: Dict[str, Any], websocket, llm_refiner: Callable):
        self.graph_def = graph_def
        self.websocket = websocket
        self.llm_refiner = llm_refiner
        self.state_graph = None

    def extract_terminal_nodes(self) -> List[str]:
        target_ids = {edge['target'] for edge in self.graph_def['edges']}
        source_ids = {edge['source'] for edge in self.graph_def['edges']}
        return list(target_ids - source_ids)

    def extract_entry_nodes(self) -> List[str]:
        target_ids = {edge['target'] for edge in self.graph_def['edges']}
        source_ids = {edge['source'] for edge in self.graph_def['edges']}
        return list(source_ids - target_ids)

    def build_graph(self):
        from langgraph.graph import StateGraph
        from langchain_core.runnables import RunnableLambda

        graph = StateGraph(state_schema=dict)
        node_map = {}

        for node in self.graph_def['nodes']:
            node_id = node['id']
            node_map[node_id] = RunnableLambda(self.make_node_runner(node))
            graph.add_node(node_id, node_map[node_id])

        for edge in self.graph_def['edges']:
            graph.add_edge(edge['source'], edge['target'])

        entry_nodes = self.extract_entry_nodes()
        if entry_nodes:
            graph.set_entry_point(entry_nodes[0])  # Assuming single entry

        terminal_nodes = self.extract_terminal_nodes()
        for t in terminal_nodes:
            graph.set_finish_point(t)

        self.state_graph = graph.compile()

    def make_node_runner(self, node):
        async def run_node(state: Dict[str, Any]):
            method = node['method']
            operation_id = node['operation_id']
            endpoint = node['endpoint']

            payload = await self.generate_payload(method, operation_id, endpoint)

            await self.websocket.send_json({
                "type": "executing_api",
                "message": f"Executing {method.upper()} {endpoint} ({operation_id})",
                "method": method,
                "endpoint": endpoint,
                "operation_id": operation_id,
                "payload": payload,
            })

            # Simulated response
            result = {"id": "123", "status": "success", "data": {"example": True}}

            await self.websocket.send_json({
                "type": "api_response",
                "message": f"Executed {method.upper()} {endpoint}. Response: {json.dumps(result)}",
                "operation_id": operation_id,
                "response": result
            })

            return {**state, operation_id: result}

        return run_node

    async def generate_payload(self, method: str, operation_id: str, endpoint: str) -> Dict[str, Any]:
        payload = {"name": "default", "price": 100}
        return await self.confirm_payload(method, operation_id, endpoint, payload)

    async def confirm_payload(self, method: str, operation_id: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        current_payload = payload

        await self.websocket.send_json({
            "type": "confirm_payload",
            "method": method,
            "operation_id": operation_id,
            "endpoint": endpoint,
            "payload": current_payload,
            "message": (
                "Here's the initial payload. You can say things like:\n"
                "- change name to 'abc'\n"
                "- remove 'price'\n"
                "- looks fine now"
            )
        })

        while True:
            user_msg = await self.websocket.receive_text()
            result = await self.llm_refiner(current_payload, user_msg)

            if isinstance(result, dict):
                current_payload = result
                await self.websocket.send_json({
                    "type": "confirm_payload",
                    "method": method,
                    "operation_id": operation_id,
                    "endpoint": endpoint,
                    "payload": current_payload,
                    "message": "Updated payload. Confirm or provide more changes."
                })
            elif result == "__approved__":
                await self.websocket.send_json({
                    "type": "confirmation_done",
                    "method": method,
                    "operation_id": operation_id,
                    "endpoint": endpoint,
                    "payload": current_payload,
                    "message": "Payload confirmed and ready to proceed."
                })
                return current_payload
            else:
                await self.websocket.send_json({
                    "type": "confirm_payload",
                    "method": method,
                    "operation_id": operation_id,
                    "endpoint": endpoint,
                    "payload": current_payload,
                    "message": "Didn't understand that. Please confirm or modify the payload."
                })

    async def astream(self):
        await self.websocket.send_json({
            "type": "workflow_start",
            "message": "Building API execution workflow..."
        })

        async for state in self.state_graph.astream({}):
            await self.websocket.send_json({
                "type": "state_update",
                "message": f"Current state updated: {json.dumps(state)}",
                "state": state
            })

        await self.websocket.send_json({
            "type": "workflow_complete",
            "message": "API execution workflow completed successfully."
        })


async def llm_refiner(payload: Dict[str, Any], user_instruction: str) -> Any:
    prompt = f"""
You're an assistant helping refine API JSON payloads based on user instructions.

Current Payload:
{json.dumps(payload, indent=2)}

Instruction from user: "{user_instruction}"

Respond with one of the following:
1. If the instruction is approval (like "yes", "looks fine", "approved"), respond with: __approved__
2. If the instruction modifies the payload, return the modified JSON only.

Respond with valid JSON or "__approved__".
"""

    response = await fake_openai_call(prompt)
    response = response.strip()

    if response == "__approved__":
        return "__approved__"

    try:
        return json.loads(response)
    except Exception:
        return None


async def fake_openai_call(prompt: str) -> str:
    print("LLM Prompt:", prompt)
    return "__approved__"




import asyncio
from langgraph.graph import Command
from langgraph.checkpoint.memory import MemorySaver

async def astream(self, initial_state=None, config=None):
    # Prepare state and config
    state_dict = (initial_state or GraphState()).dict()
    cfg = config or {}
    if "thread_id" not in cfg:
        raise ValueError("Missing thread_id in config for checkpointing")

    # Create the async generator once
    gen = self.runner.astream(state_dict, cfg, stream_mode=["values","updates"])  # streaming setup&#8203;:contentReference[oaicite:5]{index=5}

    # Outer loop to allow repeated interrupts
    while True:
        # Drive generator until exhaust or interrupt
        async for stream_mode, step in gen:
            # ---- human-in-the-loop interrupt ----
            if stream_mode == "updates" and "_interrupt_" in step:
                intr = step["_interrupt_"][0]
                # send payload confirmation to UI
                await self.websocket_callback(
                    "payload_confirmation",
                    {
                        "interrupt_key": intr.ns[0],
                        "prompt": intr.value.get("question"),
                        "payload": intr.value.get("payload", {}),
                    },
                )  # interrupt messaging&#8203;:contentReference[oaicite:6]{index=6}

                # wait for user resume, with timeout to avoid deadlock
                try:
                    resume_value = await asyncio.wait_for(
                        self.resume_queue.get(),
                        timeout=60  # seconds&#8203;:contentReference[oaicite:7]{index=7}
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError("User did not resume in time")  # fail-fast

                # resume the graph from this interrupt
                gen = self.runner.astream(
                    Command(resume=resume_value),
                    cfg,
                    stream_mode=["values","updates"],
                )  # reassign generator on resume&#8203;:contentReference[oaicite:8]{index=8}

                break  # break async for → restart outer while

            # ---- normal values processing ----
            elif stream_mode == "values":
                ops = step.get("operations", [])
                if not ops:
                    continue
                op = ops[-1]
                api_name = op["operationId"]
                # … your metrics, logging, websocket updates …

        else:
            # async for exhausted normally → graph done → exit loop&#8203;:contentReference[oaicite:9]{index=9}
            return

        # if we broke due to interrupt, loop repeats; else we would have returned
