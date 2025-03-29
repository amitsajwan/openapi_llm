import json
import logging
from fastapi import FastAPI, WebSocket
from openai import AzureChatOpenAI  # Import Azure Chat OpenAI
from openapi_parser import OpenAPIParser
from api_executor import APIExecutor
from api_workflow import APIWorkflowManager
from llm_sequence_generator import LLMSequenceGenerator
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize AzureChatOpenAI
llm_client = AzureChatOpenAI()


app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Store OpenAPI Spec Data
openapi_data = None

# # Initialize Azure OpenAI Client
# llm_client = AzureChatOpenAI(deployment_name="your-deployment-name", api_key="your-api-key", api_version="2023-03-15-preview")


def load_openapi_from_url_or_file(source: str):
    global openapi_data
    parser = OpenAPIParser()
    openapi_data = parser.parse(source)
    return openapi_data


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Welcome! Please provide the OpenAPI (Swagger) URL or upload a spec file.")
    llm = LLMSequenceGenerator(llm_client)  # Use AzureChatOpenAI
    
    while True:
        user_input = await websocket.receive_text()
        intent = llm.determine_intent(user_input, openapi_data)
        
        if intent == "provide_openapi":
            openapi_data = load_openapi_from_url_or_file(user_input)
            await websocket.send_text("OpenAPI Spec Loaded! You can ask about available APIs or run tests.")
        
        elif intent == "list_apis":
            apis = json.dumps(openapi_data.get_endpoints(), indent=2)
            await websocket.send_text(f"Available APIs:\n{apis}")
        
        elif intent == "run_sequence":
            sequence = llm.suggest_sequence(openapi_data)
            await websocket.send_text(f"Suggested Execution Sequence: {sequence}. Confirm?")
            confirmation = await websocket.receive_text()
            if "yes" in confirmation.lower():
                workflow_manager = APIWorkflowManager(openapi_data, llm_client)
                result = await workflow_manager.execute_workflow(sequence)
                await websocket.send_text(f"Execution Results: {result}")
        
        elif intent == "load_test":
            num_users, duration = llm.extract_load_test_params(user_input)
            executor = APIExecutor()
            results = await executor.run_load_test(openapi_data, num_users, duration)
            await websocket.send_text(f"Load Test Results:\n{results}")
        
        elif intent == "general_query":
            response = openapi_data.answer_query(user_input)
            await websocket.send_text(response)
        
        elif intent == "execute_api":
            method, endpoint = llm.extract_api_details(user_input)
            executor = APIExecutor()
            payload = llm.generate_payload(endpoint, openapi_data)
            result = await executor.execute_api(method, endpoint, payload)
            await websocket.send_text(f"Execution Result: {result}")
        
        elif intent == "modify_execution":
            await websocket.send_text("Would you like to modify the execution sequence? Provide new order.")
            new_sequence = await websocket.receive_text()
            workflow_manager = APIWorkflowManager(openapi_data, llm_client)
            modified_result = await workflow_manager.execute_workflow(json.loads(new_sequence))
            await websocket.send_text(f"Modified Execution Results: {modified_result}")
        
        else:
            await websocket.send_text("I didn't understand. Try asking about APIs, running tests, or modifying the execution sequence.")
