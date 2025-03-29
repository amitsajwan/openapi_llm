from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import AzureChatOpenAI
from llm_sequence_generator import LLMSequenceGenerator  # Ensure this is correctly imported

import requests
import json
import yaml

# FastAPI app initialization
app = FastAPI()

# Mount static files (CSS, JS) for serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the LLM (AzureChatOpenAI)
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # Use your deployment name
    api_version="2023-06-01-preview",  # Use your API version
    temperature=0.5,
    max_tokens=1000,
    timeout=30,
    max_retries=3
)

# Initialize LLMSequenceGenerator with AzureChatOpenAI instance
llm_sequence_generator = LLMSequenceGenerator(llm_client=llm)

# Global variable to hold OpenAPI data
openapi_data = {}

# Route to load OpenAPI from URL or file
@app.post("/load_openapi")
async def load_openapi(url: str = Form(None), file: UploadFile = File(None)):
    global openapi_data

    # If URL is provided
    if url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                openapi_data = response.json()  # Assuming the URL returns JSON
                return JSONResponse(content={"message": "OpenAPI data loaded from URL."})
            else:
                raise HTTPException(status_code=400, detail="Unable to fetch data from the provided URL.")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail="Error fetching data from the URL.")

    # If file is provided
    if file:
        try:
            content = await file.read()
            try:
                openapi_data = json.loads(content)  # Try to parse as JSON
            except json.JSONDecodeError:
                try:
                    openapi_data = yaml.safe_load(content)  # Try parsing as YAML
                except yaml.YAMLError:
                    raise HTTPException(status_code=400, detail="Invalid file format. Must be JSON or YAML.")
            return JSONResponse(content={"message": "OpenAPI data loaded from file."})
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error reading the file.")

    # If neither URL nor file is provided
    raise HTTPException(status_code=400, detail="Please provide a Swagger URL or file.")

# Route for the user to interact with LLM (e.g., asking questions, suggesting API sequence)
@app.post("/interact_with_llm")
async def interact_with_llm(user_input: str):
    global openapi_data

    if not openapi_data:
        raise HTTPException(status_code=400, detail="No OpenAPI data loaded.")

    try:
        intent = await llm_sequence_generator.determine_intent(user_input, openapi_data)
        return JSONResponse(content={"intent": intent})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static files serving (index.html)
@app.get("/")
async def serve_ui():
    return JSONResponse(content={"message": "UI served. Visit /static/index.html for the app."})

