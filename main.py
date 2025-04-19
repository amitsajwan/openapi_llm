import asyncio
import uuid
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langgraph_builder import LangGraphBuilder, GraphState
from api_executor import APIExecutor

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        # 1) Receive graph JSON
        raw = await ws.receive_text()
        graph_def = json.loads(raw)

        # 2) Instantiate API executor & builder
        api_exec = APIExecutor(base_url="https://api.example.com")
        builder = LangGraphBuilder(
            graph_def=graph_def,
            api_executor=api_exec,
            websocket_callback=lambda t, p: asyncio.create_task(ws.send_json({"type": t, **p}))
        )

        # 3) Setup state + config
        state = GraphState()
        config = {"thread_id": uuid.uuid4().hex}

        # 4) Run astream & receiver concurrently
        async def sender():
            async for _ in builder.astream(state, config):
                pass

        async def receiver():
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "user_payload_confirmation":
                    # pass the user's payload back into the builder
                    await builder.submit_resume(msg["payload"])

        await asyncio.gather(sender(), receiver())

    except WebSocketDisconnect:
        return
