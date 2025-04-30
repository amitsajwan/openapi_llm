# fastapi_app.py
import uvicorn
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional

# Assume models.py, tools.py, and openapi_agent.py are in the same directory
from models import BotState, BotIntent, GraphOutput, TextContent # Import necessary models
from openapi_agent import LangGraphOpenApiRouter # Import the agent class
from tools import MockLLM # Import MockLLM (replace with your actual LLM imports)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Static Files (Optional: for serving a simple HTML client) ---
# Mount a directory named "static" to serve files like index.html
# Create a directory named 'static' in the same location as fastapi_app.py
# and put an index.html file (example provided below) inside it.
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- LLM and Agent Initialization ---
# IMPORTANT: Replace MockLLM with your actual LangChain LLM instances
# These should be initialized once when the app starts.
llm_router = MockLLM(role="router")
llm_worker = MockLLM(role="worker")

# Define the OpenAPI Spec (Example) - Load this from a file in a real app
openapi_spec_content = """
openapi: 3.0.0
info:
  title: Simple API
  version: 1.0.0
paths:
  /login:
    post:
      operationId: loginUser
      summary: User Login
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                username: { type: string }
                password: { type: string }
  /profile:
    get:
      operationId: getUserProfile
      summary: Get User Profile
      parameters:
        - name: Authorization
          in: header
          required: true
          schema: { type: string }
  /items:
    get:
      operationId: listItems
      summary: List Items
      parameters:
        - name: Authorization
          in: header
          required: true
          schema: { type: string }
  /items/{id}:
    get:
      operationId: getItemDetails
      summary: Get Item Details
      parameters:
        - name: id
          in: path
          required: true
          schema: { type: integer }
        - name: Authorization
          in: header
          required: true
          schema: { type: string }
"""

# In a real application, you might load the spec dynamically or based on user input
# For this example, the agent is initialized with a fixed spec content.
# The session_id will manage state for different WebSocket connections.
# The spec_text is passed during initialization, and the agent's load_state will
# handle whether it needs to be parsed or is already in the cached state.
# A simple approach for multiple users/specs could involve initializing
# an agent *per user session* or having a mechanism to switch specs within the agent.
# For this example, we'll assume a single agent instance handles multiple sessions
# with the *same* initial spec content, relying on session state for conversation history.
# A more complex app might have a map of session_id -> agent instance.

# Initialize the agent instance (outside the endpoint)
# We'll use a placeholder session_id "initial_load" for the agent's internal init,
# but actual session management happens per WebSocket connection.
# The spec_text is provided here for the agent to potentially parse on the first run
# of any session or if a session's state is not found/corrupted.
agent_instance = LangGraphOpenApiRouter(
    session_id="initial_load", # Placeholder ID for agent initialization phase
    spec_text=openapi_spec_content,
    llm_router=llm_router,
    llm_worker=llm_worker,
    cache_dir="./agent_cache" # Directory to save/load session state
)

logger.info("LangGraphOpenApiRouter agent instance created.")


# --- WebSocket Endpoint ---
# The client is expected to send JSON messages with at least a "user_input" field
# and optionally a "session_id" field to maintain state.
# If no session_id is sent, a new random one will be generated for that connection.

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    # Simple session management: use a session ID provided by the client
    # or generate a new one if not provided.
    # In a real app, you'd have more robust session management.
    session_id: Optional[str] = None

    try:
        while True:
            # Receive message from client (expecting JSON string)
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")

            try:
                message = json.loads(data)
                user_input = message.get("user_input")
                client_session_id = message.get("session_id") # Get session ID from client

                if not user_input:
                    await websocket.send_text(json.dumps({"error": "Missing 'user_input' in message."}))
                    continue

                # Determine the session ID to use for this interaction
                if client_session_id:
                    session_id = client_session_id
                    logger.info(f"Using client-provided session ID: {session_id}")
                elif session_id is None:
                    # Generate a new session ID if this is the first message and none was provided
                    session_id = os.urandom(16).hex()
                    logger.info(f"Generated new session ID: {session_id}")

                # --- Invoke the Agent ---
                # The agent handles loading/saving state based on the session_id
                updated_state: BotState = agent_instance.invoke_graph(
                    user_input=user_input,
                    session_id=session_id
                )

                # --- Prepare Response ---
                # Extract text response
                bot_response_text = updated_state.text_content.text

                # Extract graph data if available
                graph_data_json: Optional[Dict[str, Any]] = None
                if updated_state.graph_output:
                    try:
                        # Convert Pydantic model to dictionary
                        graph_data_json = updated_state.graph_output.model_dump()
                        # You might want to clean up None values if sending over JSON
                        # graph_data_json = json.loads(updated_state.graph_output.model_dump_json()) # Ensure it's pure JSON serializable
                    except Exception as e:
                        logger.error(f"Error serializing graph output: {e}")
                        # Optionally send an error about graph serialization
                        pass # Continue without graph data if serialization fails


                # Construct the response message (JSON)
                response_payload = {
                    "session_id": session_id, # Send the session ID back to the client
                    "response_text": bot_response_text,
                    "graph_output": graph_data_json, # Include graph data if available
                    "intent": updated_state.intent.value, # Optionally include the determined intent
                    # You could add other state info here if useful for the client
                    # "action_history": updated_state.action_history,
                    # "scratchpad_reason": updated_state.scratchpad.get("reason", "") # Be cautious about sending full scratchpad
                }

                # Send the JSON response back to the client
                await websocket.send_text(json.dumps(response_payload))
                logger.info(f"Sent response for session {session_id}.")

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format."}))
                logger.warning("Received non-JSON message.")
            except Exception as e:
                logger.error(f"An error occurred during message processing: {e}", exc_info=True)
                await websocket.send_text(json.dumps({"error": f"An internal error occurred: {e}"}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection disconnected (Session ID: {session_id}).")
    except Exception as e:
        logger.error(f"An unexpected error occurred in WebSocket: {e}", exc_info=True)


# --- Root Endpoint (Optional: serves the index.html client) ---
# This is just for easily testing the WebSocket endpoint with a browser.
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Reads the index.html file from the 'static' directory
    # Make sure you have a 'static' directory with index.html
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>FastAPI OpenAPI Bot</h1><p>WebSocket endpoint is at /ws. Create a 'static' directory with an index.html file to serve a client.</p>", status_code=404)


# --- Running the App ---
# To run this application:
# 1. Save the code as fastapi_app.py
# 2. Make sure models.py, tools.py, and openapi_agent.py are in the same directory.
# 3. Create a 'static' directory and add an index.html file (see example below).
# 4. Install uvicorn: pip install uvicorn
# 5. Run from your terminal: uvicorn fastapi_app:app --reload
# 6. Open your browser to http://127.0.0.1:8000/ (if you created index.html)
#    or use a WebSocket client to connect to ws://127.0.0.1:8000/ws

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

