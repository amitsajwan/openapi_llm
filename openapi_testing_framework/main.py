# main.py

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

# Simple HTML page with WebSocket-based chat UI
html = """
<!DOCTYPE html>
<html>
<head>
  <title>OpenAPI Test Framework</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2em; }
    #messages { list-style: none; padding: 0; }
    #messages li { margin-bottom: 0.5em; }
    #input { width: 80%; padding: 0.5em; }
    #send { padding: 0.5em; }
  </style>
</head>
<body>
  <h1>OpenAPI Test Framework</h1>
  <ul id="messages"></ul>
  <input id="input" autocomplete="off" placeholder="Type a command…" />
  <button id="send">Send</button>
  <script>
    const ws = new WebSocket(`ws://${location.host}/ws`);
    const msgs = document.getElementById('messages');
    const input = document.getElementById('input');
    const send = document.getElementById('send');

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const li = document.createElement('li');
      li.textContent = `${msg.node}: ${JSON.stringify(msg.output)}`;
      msgs.appendChild(li);
      window.scrollTo(0, document.body.scrollHeight);
    };

    send.onclick = () => {
      if (input.value) {
        ws.send(input.value);
        input.value = '';
      }
    };

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        send.click();
      }
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = ws.headers.get('sec-websocket-key')

    # Load persisted state or start fresh
    state_dict = load_state(GRAPH_NAME, session_id)
    state = BotState(**state_dict)

    # Build and compile the graph once per session
    graph = build_openapi_graph()

    # If no spec chosen yet, prompt user
    if not state.spec_path:
        await ws.send_json({"node": "system", "output": {"available_specs": AVAILABLE_SPECS}})
        await ws.send_json({"node": "system", "output": "Please type the filename of the OpenAPI spec to load."})

    try:
        while True:
            user_msg = await ws.receive_text()
            state.user_input = user_msg

            # Stream execution through the graph
            stream = graph.astream(
                inputs={"state": state, "user_input": user_msg},
                session_id=session_id
            )
            async for event in stream:
                await ws.send_json({"node": event.node_name, "output": event.output})

            # Persist updated state after each full run
            save_state(state.dict(), GRAPH_NAME, session_id)

    except WebSocketDisconnect:
        # Client disconnected gracefully
        pass
