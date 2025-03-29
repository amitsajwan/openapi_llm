import json
import os
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_openai import AzureChatOpenAI
from LLMSequenceGenerator import LLMSequenceGenerator

app = FastAPI()

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Azure OpenAI properties from environment variables
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-35-turbo"),
    api_version=os.getenv("AZURE_API_VERSION", "2023-06-01-preview"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_generator = LLMSequenceGenerator(llm)

openapi_data = None  # Store OpenAPI data globally

@app.post("/upload_openapi")
async def upload_openapi(file: UploadFile = File(...)):
    global openapi_data
    try:
        openapi_data = json.loads(await file.read())
        return JSONResponse(content={"message": "OpenAPI file uploaded successfully."})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

@app.post("/upload_openapi_url")
async def upload_openapi_url(url: str = Form(...)):
    global openapi_data
    try:
        # Simulate fetching JSON from URL (replace with real HTTP request in production)
        openapi_data = {"mocked": "This would be fetched from the URL"}  # Replace with actual fetching logic
        return JSONResponse(content={"message": "OpenAPI schema loaded from URL."})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/determine_intent")
async def determine_intent(user_input: str = Form(...)):
    if not openapi_data:
        raise HTTPException(status_code=400, detail="OpenAPI data not uploaded.")
    response = await llm_generator.determine_intent(user_input, openapi_data)
    return JSONResponse(content={"intent": response})

@app.post("/suggest_sequence")
async def suggest_sequence():
    if not openapi_data:
        raise HTTPException(status_code=400, detail="OpenAPI data not uploaded.")
    response = await llm_generator.suggest_sequence(openapi_data)
    return JSONResponse(content={"sequence": response})

@app.post("/extract_load_test_params")
async def extract_load_test_params(user_input: str = Form(...)):
    response = await llm_generator.extract_load_test_params(user_input)
    return JSONResponse(content={"num_users": response[0], "duration": response[1]})

@app.post("/extract_api_details")
async def extract_api_details(user_input: str = Form(...)):
    response = await llm_generator.extract_api_details(user_input)
    return JSONResponse(content={"method": response[0], "endpoint": response[1]})

@app.post("/generate_payload")
async def generate_payload(endpoint: str = Form(...)):
    if not openapi_data:
        raise HTTPException(status_code=400, detail="OpenAPI data not uploaded.")
    response = await llm_generator.generate_payload(endpoint, openapi_data)
    return JSONResponse(content={"payload": response})

@app.get("/")
async def root():
    return JSONResponse(content={"message": "API is running. Upload OpenAPI data and interact with LLM."})
