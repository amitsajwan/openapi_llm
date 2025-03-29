import json
import os
from fastapi import FastAPI, WebSocket
from langchain_openai import AzureChatOpenAI
import requests
from llm_sequence_generator import LLMSequenceGenerator

# Load environment variables
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-35-turbo")  # Default to gpt-35-turbo if not provided

app = FastAPI()

# Initialize Azure Chat OpenAI
llm = AzureChatOpenAI(
    azure_deployment=azure_deployment,
    api_version="2023-06-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=azure_api_key,
    endpoint=azure_endpoint
)

llm_sequence_generator = LLMSequenceGenerator(llm_client=llm)

openapi_data = None  # Initialize OpenAPI data

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Welcome! Please provide the OpenAPI (Swagger) URL or upload a spec file.")
    
    while True:
        user_input = await websocket.receive_text()
        
        if "openapi" in user_input:
            # Load OpenAPI from URL or file (as before)
            openapi_data = load_openapi(user_input)
            await websocket.send_text(f"OpenAPI Spec Loaded: {json.dumps(openapi_data)}")
        else:
            # Process user query with LLMSequenceGenerator
            response = await llm_sequence_generator.answer_general_query(user_input, openapi_data)
            await websocket.send_text(response)

def load_openapi(source: str):
    """Load OpenAPI specification from a URL or file."""
    if source.startswith("http"):  # If it's a URL, download the OpenAPI spec
        response = requests.get(source)
        return response.json()
    else:  # If it's a file path, read the OpenAPI spec
        with open(source, 'r') as file:
            return json.load(file)


from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Existing FastAPI app and WebSocket code...

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html
@app.get("/")
async def serve_index():
    return FileResponse("index.html")
