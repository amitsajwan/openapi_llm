"""
main.py

Entrypoint for the OpenAPI testing framework:
  - Sets up FastAPI app with WebSocket endpoint for chat-based interaction.
  - On connect: loads or initializes BotState, loads available specs.
  - On each message: passes user input into the LangGraph StateGraph, streams node outputs back over WebSocket.
  - Persists state via a dedicated LangGraph saver node rather than in main.
  - Differentiates between text messages (prompts/status) and JSON messages (graph/tool outputs).
"""
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from graph import build_openapi_graph
from state_manager import load_state, save_state
from models import BotState
from utils import read_swagger_files

app = FastAPI()
GRAPH_NAME = "openapi_test_flow"

# load available OpenAPI spec files once
SPEC_FOLDER = "./specs"
AVAILABLE_SPECS = read_swagger_files(SPEC_FOLDER)

html = """
<!DOCTYPE html>
<html>
  <head><title>API Test Chat</title></head>
  <body>
    <h1>OpenAPI Test Chat</h1>
    <textarea id="log" cols="80" rows="20" readonly></textarea><br>
    <input id="input" type="text" size="80" placeholder="Type a command..." />
    <button onclick="send()">Send</button>
    <script>
      var ws = new WebSocket("ws://" + location.host + "/ws");
      var log = document.getElementById('log');
      ws.onmessage = function(event) {
        if (event.data.startsWith('{')) {
          // JSON message: graph/tool output
          console.log("Graph JSON:", JSON.parse(event.data));
        } else {
          // Plain text message
          log.value += event.data + '\n';
        }
      };
      function send() {
        var input = document.getElementById('input');
        ws.send(input.value);
        input.value = '';
      }
    </script>
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.headers.get('sec-websocket-key')

    # initialize or load state
    state_dict = load_state(GRAPH_NAME, session_id)
    state = BotState(**state_dict)

    # build graph
    graph = build_openapi_graph()

    try:
        # greet and prompt for spec selection if first time
        if not state.spec_path:
            await websocket.send_text(f"Available specs: {AVAILABLE_SPECS}")
            await websocket.send_text("Please type the file path of the spec to load.")

        while True:
            user_input = await websocket.receive_text()
            # run the graph with the latest state and user_input
            result = graph.run(
                inputs={"state": state, "user_input": user_input},
                session_id=session_id,
                stream=True
            )

            async for event in result:
                # send JSON payload for tool outputs
                payload = event.output if isinstance(event.output, dict) else json.loads(json.dumps(event.output))
                await websocket.send_json({"node": event.node_name, "output": payload})

            # after the full graph run, persist state once
            save_state(state.dict(), GRAPH_NAME, session_id)

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
