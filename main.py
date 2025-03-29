import json
import os
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from LLMSequenceGenerator import LLMSequenceGenerator  # Assuming you have this in a separate file

# Load environment variables from .env file
load_dotenv()

# Initialize the AzureChatOpenAI client
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your API version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize LLMSequenceGenerator
sequence_generator = LLMSequenceGenerator(llm)

# Create FastAPI instance
app = FastAPI()

# Serve static files (for the HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for OpenAPI data
app.state.openapi_data = None

# Pydantic model for user input
class UserInputRequest(BaseModel):
    user_input: str


@app.post("/upload_openapi_url")
async def upload_openapi_url(url: str):
    """
    Handle OpenAPI URL input and fetch the OpenAPI spec from the URL.
    """
    global openapi_data
    try:
        # Fetch OpenAPI spec from URL
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the OpenAPI JSON data from URL
            openapi_data = response.json()
            return JSONResponse(content={"message": "OpenAPI loaded from URL successfully."}, status_code=200)
        else:
            raise HTTPException(status_code=422, detail="Failed to fetch OpenAPI from URL")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching OpenAPI: {str(e)}")


# Endpoint to upload OpenAPI data (JSON file)
@app.post("/upload_openapi")
async def upload_openapi(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content
        openapi_content = await file.read()
        
        # Parse the JSON data from the file
        openapi_json = json.loads(openapi_content.decode("utf-8"))
        
        # Store OpenAPI data in the app's state
        app.state.openapi_data = openapi_json
        
        return {"message": "OpenAPI file uploaded successfully", "data": openapi_json}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading OpenAPI file: {str(e)}")

# Endpoint to submit OpenAPI data and interact with LLM
@app.post("/submit_openapi")
async def submit_openapi(user_input: UserInputRequest):
    try:
        # Check if OpenAPI data is available
        openapi_data = app.state.openapi_data
        if not openapi_data:
            raise HTTPException(status_code=400, detail="No OpenAPI data available. Please upload OpenAPI data first.")
        
        # Ensure user input is provided
        if not user_input.user_input:
            raise HTTPException(status_code=422, detail="User input cannot be empty.")
        
        # Use LLM to process the input and provide a response using LLMSequenceGenerator
        response = await sequence_generator.answer_general_query(user_input.user_input, openapi_data)
        
        # Return LLM's response
        return {"message": "LLM response", "response": response}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Serve the UI page (HTML form and chatbot interface)
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")
