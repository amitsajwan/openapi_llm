import json
import os
import requests
import yaml
from langchain_openai import AzureChatOpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio

# Load Azure environment variables
from dotenv import load_dotenv

load_dotenv()  # Ensure .env file is loaded

# Initialize FastAPI app
app = FastAPI()

# Initialize the AzureChatOpenAI client using environment variables
llm_client = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_API_KEY"),
    deployment_name="your_deployment_name",  # Replace with your deployment name
    endpoint=os.getenv("AZURE_API_ENDPOINT")
)

# Initialize LLMSequenceGenerator
class LLMSequenceGenerator:
    def __init__(self, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client

    async def determine_intent(self, user_input: str, openapi_data):
        """Determines user intent based on input and OpenAPI context."""
        prompt = f"User input: {user_input}\nOpenAPI Context: {json.dumps(openapi_data, indent=2) if openapi_data else 'No OpenAPI Data'}\nIntent:"
        response = await self.llm_client.ainvoke(prompt)
        return response.content.strip()

    async def list_apis(self, openapi_data):
        """Lists available APIs."""
        paths = openapi_data.get("paths", {})
        return [path for path in paths.keys()]

# Load OpenAPI Data from URL or file
def load_openapi_data(url_or_file):
    """Load OpenAPI data either from a URL or file."""
    if url_or_file.startswith("http"):
        # Fetch OpenAPI data from a URL
        response = requests.get(url_or_file)
        response.raise_for_status()
        return response.json()  # Assuming OpenAPI is in JSON format
    else:
        # Load OpenAPI data from a file
        with open(url_or_file, 'r') as file:
            return yaml.safe_load(file)  # If the file is in YAML format

# Endpoint to serve the UI
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

# Endpoint to load OpenAPI data
@app.post("/load_openapi/")
async def load_openapi(url_or_file: str):
    try:
        openapi_data = load_openapi_data(url_or_file)
        return {"status": "success", "openapi_data": openapi_data}
    except Exception as e:
        return {"status": "failure", "message": str(e)}

# WebSocket for chat interaction
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Initialize LLM Sequence Generator
        llm_generator = LLMSequenceGenerator(llm_client)
        openapi_data = None

        while True:
            user_input = await websocket.receive_text()

            if user_input == "List available APIs":
                if openapi_data:
                    api_list = await llm_generator.list_apis(openapi_data)
                    response = f"Available APIs: {', '.join(api_list)}"
                else:
                    response = "OpenAPI data is not loaded yet."
            else:
                # Determine user intent
                intent = await llm_generator.determine_intent(user_input, openapi_data)
                response = f"Intent: {intent}"

            # Send response to user
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("Client disconnected")

# Serve static files (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
