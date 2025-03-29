import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from LLMSequenceGenerator import LLMSequenceGenerator  # Importing the LLMSequenceGenerator module
from dotenv import load_dotenv

load_dotenv()

# Initialize AzureChatOpenAI client
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Initialize the LLMSequenceGenerator with the LLM client
sequence_generator = LLMSequenceGenerator(llm)

# Define FastAPI app
app = FastAPI()

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

# Pydantic model to validate user input
class UserInputRequest(BaseModel):
    user_input: str

# Endpoint to upload OpenAPI data (JSON file)
@app.post("/upload_openapi")
async def upload_openapi(file: Request):
    try:
        # Get the file from the request
        file_data = await file.form()
        openapi_file = file_data["file"]
        
        # Parse the JSON data
        openapi_json = json.loads(openapi_file.file.read().decode("utf-8"))
        
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

# Entry point to serve the web UI (static files)
@app.get("/")
async def serve_ui():
    return {"message": "API is running. You can upload OpenAPI data and submit queries to interact with LLM."}
