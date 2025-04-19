# main.py
import asyncio, uuid, json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langgraph_builder import LangGraphBuilder, GraphState
from api_executor import APIExecutor

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # 1. Receive graph spec
    raw = await ws.receive_text()
    graph_def = json.loads(raw)

    # 2. Prepare API executor
    api_exec = APIExecutor(base_url="https://api.example.com")

    # 3. Define an async callback so send_json is awaited
    async def ws_callback(msg_type: str, payload: dict):
        await ws.send_json({"type": msg_type, **payload})

    # 4. Build the LangGraph runner
    builder = LangGraphBuilder(
        graph_def=graph_def,
        api_executor=api_exec,
        websocket_callback=ws_callback
    )
    runner = builder.runner  # already compiled with MemorySaver()

    # 5. Start streaming
    state = GraphState()
    config = {"thread_id": uuid.uuid4().hex}

    try:
        async for _ in builder.astream(state, config):
            # all messaging handled inside astream via ws_callback
            pass
    except WebSocketDisconnect:
        # client disconnected
        return
