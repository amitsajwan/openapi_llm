# main.py

import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

app = FastAPI()


# 1. Define your state schema
class GraphState(BaseModel):
    operation_results: dict = {}
    extracted_ids: dict = {}


# 2. A simple API executor stub
class APIExecutor:
    async def execute_api(self, method, endpoint, payload):
        # simulate API call...
        await asyncio.sleep(0.1)
        return {"status": 200, "data": {"payload": payload}}

    def extract_ids(self, result):
        # stub: no ids
        return {}


# 3. Build the LangGraph workflow (single node for brevity)
def build_runner():
    builder = StateGraph(GraphState)
    # A single node that interrupts
    async def human_then_api(state: GraphState) -> GraphState:
        # Prepare payload
        initial = {"question": "Please revise the text", "some_text": "Original text"}
        # Pause here: yield an Interrupt event
        reply = interrupt(initial)  # single‑arg interrupt :contentReference[oaicite:1]{index=1}

        # After resume, `reply` holds the user's dict
        # Now simulate API call with the confirmed payload
        exec_result = await APIExecutor().execute_api(
            method="POST",
            endpoint="/dummy",
            payload=reply.get("some_text")
        )
        # Return updated state
        return GraphState(
            operation_results={"human_then_api": exec_result},
            extracted_ids={}
        )

    builder.add_node("human_then_api", RunnableLambda(human_then_api))
    builder.set_entry_point("human_then_api")
    # Compile with MemorySaver for interrupt support
    return builder.compile(checkpointer=MemorySaver())


# 4. WebSocket handler: stream + resume
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    runner = build_runner()
    # Unique thread_id required by MemorySaver
    config = {"thread_id": uuid.uuid4().hex}

    try:
        # Stream through the graph
        async for event in runner.astream({}, config):  # empty initial state
            # 4a. Handle the interrupt event
            if "__interrupt__" in event:
                intr_obj = event["__interrupt__"][0]
                # intr_obj.value is {'question':..., 'some_text':...}
                await ws.send_json({
                    "type": "payload_confirmation",
                    "interrupt_key": intr_obj.ns[0],
                    "prompt": intr_obj.value["question"],
                    "payload": intr_obj.value
                })
                # Wait for user reply
                msg = await ws.receive_json()
                # Resume with the user-provided dict
                await runner.submit(Command(resume=msg["payload"]), config)  :contentReference[oaicite:2]{index=2}
                continue

            # 4b. Normal node completion
            for op_id, result in event["operation_results"].items():
                await ws.send_json({
                    "type": "api_response",
                    "operationId": op_id,
                    "result": result
                })

    except WebSocketDisconnect:
        pass
