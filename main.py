from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import requests
from langchain_openai import AzureChatOpenAI

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Serving static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Endpoint for the index page (to serve index.html)
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Initialize the AzureChatOpenAI LLM client
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the LLMSequenceGenerator
class LLMSequenceGenerator:
    def __init__(self, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client

    async def determine_intent(self, user_input: str, openapi_data):
        """Determines user intent based on input and OpenAPI context."""
        prompt = f"User input: {user_input}\nOpenAPI Context: {json.dumps(openapi_data, indent=2) if openapi_data else 'No OpenAPI Data'}\nIntent:"
        response = await self.llm_client.ainvoke(prompt)
        return response.content.strip()

    async def suggest_sequence(self, openapi_data):
        """Suggests an API execution sequence."""
        prompt = f"Given the following OpenAPI data, suggest a sequence of API calls:\n{json.dumps(openapi_data, indent=2)}"
        response = await self.llm_client.ainvoke(prompt)
        return json.loads(response.content)

    async def answer_general_query(self, user_input: str, openapi_data):
        """Answers general user queries based on OpenAPI data."""
        prompt = f"User query: {user_input}\nOpenAPI Context: {json.dumps(openapi_data, indent=2)}\nAnswer:"
        response = await self.llm_client.ainvoke(prompt)
        return response.content.strip()

    async def extract_load_test_params(self, user_input: str):
        """Extracts number of users and duration for load testing."""
        prompt = f"Extract the number of users and duration from: {user_input}"
        response = await self.llm_client.ainvoke(prompt)
        params = json.loads(response.content)
        return params.get("num_users", 1), params.get("duration", 60)

    async def extract_api_details(self, user_input: str):
        """Extracts API method and endpoint from user input."""
        prompt = f"Extract API method and endpoint from: {user_input}"
        response = await self.llm_client.ainvoke(prompt)
        details = json.loads(response.content)
        return details.get("method"), details.get("endpoint")

    async def generate_payload(self, endpoint: str, openapi_data):
        """Generates a payload based on the OpenAPI specification."""
        prompt = f"Generate an example payload for the endpoint {endpoint} based on the following OpenAPI data:\n{json.dumps(openapi_data, indent=2)}"
        response = await self.llm_client.ainvoke(prompt)
        return json.loads(response.content)

# Endpoint for uploading OpenAPI file
@app.post("/upload_openapi")
async def upload_openapi(file: Request):
    try:
        file_data = await file.form()
        openapi_file = file_data["file"]
        openapi_json = json.loads(openapi_file.file.read().decode("utf-8"))
        # Store the OpenAPI data globally or in a session for further usage
        app.state.openapi_data = openapi_json
        return JSONResponse(content={"message": "OpenAPI file uploaded successfully", "data": openapi_json}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"Error uploading OpenAPI file: {str(e)}"}, status_code=400)

# Endpoint for loading OpenAPI from a URL
@app.get("/load_openapi")
async def load_openapi(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            openapi_data = response.json()
            # Store the OpenAPI data globally or in a session for further usage
            app.state.openapi_data = openapi_data
            return JSONResponse(content={"message": "OpenAPI loaded successfully", "data": openapi_data}, status_code=200)
        else:
            return JSONResponse(content={"message": "Failed to load OpenAPI from URL"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"message": f"Error loading OpenAPI from URL: {str(e)}"}, status_code=400)

# Submit OpenAPI data for LLM interaction
@app.post("/submit_openapi")
async def submit_openapi(user_input: str):
    try:
        # Retrieve the OpenAPI data from the app state
        openapi_data = app.state.openapi_data
        if not openapi_data:
            return JSONResponse(content={"message": "No OpenAPI data available. Please upload or load OpenAPI data first."}, status_code=400)

        # Initialize LLMSequenceGenerator with the Azure Chat OpenAI client
        llm_sequence_generator = LLMSequenceGenerator(llm)

        # Use LLM to process the user input and OpenAPI data
        response = await llm_sequence_generator.answer_general_query(user_input, openapi_data)

        return JSONResponse(content={"message": "LLM response", "response": response}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"message": f"Error submitting OpenAPI: {str(e)}"}, status_code=400)

# WebSocket for chat interaction
@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    # Connection message
    await websocket.send_text("Welcome to the API chat. You can start asking questions!")

    try:
        while True:
            # Receive the user input
            user_input = await websocket.receive_text()
            
            # Process the input by submitting it to the LLM (this is just a placeholder, you should call submit_openapi here)
            response = f"Received input: {user_input}"  
            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("Client disconnected")
