# main.py

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Optional
from langgraph_workflow import LangGraphWorkflow, GraphState, GraphOutput
from api_executor import APIExecutor

app = FastAPI()

@app.websocket("/chat")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        # 1) Receive initial graph definition
        init_raw = await ws.receive_text()
        graph_def = GraphOutput.parse_raw(init_raw)

        # 2) Instantiate API executor & workflow
        api_exec = APIExecutor(base_url="https://api.example.com")
        workflow = LangGraphWorkflow(
            graph_def=graph_def.dict(),
            api_executor=api_exec,
            websocket_callback=lambda t, payload: ws.send_json({"type": t, **payload})
        )

        # 3) Create tasks: one to send events, one to receive interrupts
        async def send_events():
            # Initial empty state
            initial_state = GraphState()
            # astream handles all outgoing messages (interrupts & api_results)
            async for _ in workflow.astream(initial_state):
                pass  # nothing needed here

        async def receive_responses():
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                if data.get("type") == "user_payload_confirmation":
                    # Resume the interrupted node
                    key = data["interrupt_key"]
                    value = data["payload"]
                    await workflow.submit_interrupt_response(key, value)

        # 4) Run both concurrently
        await asyncio.gather(send_events(), receive_responses())

    except WebSocketDisconnect:
        # Client disconnected — both tasks will be cancelled automatically
        return
