import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from graph import build_openapi_graph
from state_manager import load_state, save_state
from models import BotState
from utils import read_swagger_files

app = FastAPI()
GRAPH_NAME = "openapi_test_flow"
SPEC_FOLDER = "./specs"
AVAILABLE_SPECS = read_swagger_files(SPEC_FOLDER)

html = "<!DOCTYPE html><html><head><title>Tester</title></head><body><h1>OpenAPI Tester</h1><input id='i'/><button onclick='send()'>Send</button><ul id='msgs'></ul><script>var ws=new WebSocket('ws://'+location.host+'/ws');ws.onmessage=function(e){let m=JSON.parse(e.data);let li=document.createElement('li');li.textContent=JSON.stringify(m);document.getElementById('msgs').appendChild(li);};function send(){ws.send(document.getElementById('i').value);}</script></body></html>"

@app.get("/", response_class=HTMLResponse)
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    sid = ws.headers.get('sec-websocket-key')
    state_dict = load_state(GRAPH_NAME, sid)
    state = BotState(**state_dict)
    graph = build_openapi_graph()
    if not state.spec_path:
        await ws.send_text(json.dumps({"available_specs": AVAILABLE_SPECS}))
    try:
        while True:
            u = await ws.receive_text()
            stream = graph.astream(inputs={"state": state, "user_input": u}, session_id=sid)
            async for evt in stream:
                await ws.send_text(json.dumps({"node": evt.node_name, "output": evt.output}))
            save_state(state.dict(), GRAPH_NAME, sid)
    except WebSocketDisconnect:
        pass
