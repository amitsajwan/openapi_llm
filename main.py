import json
import logging
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AzureChatOpenAI  # Import Azure Chat OpenAI
from openapi_parser import OpenAPIParser
from api_executor import APIExecutor
from api_workflow import APIWorkflowManager
from llm_sequence_generator import LLMSequenceGenerator

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Store OpenAPI Spec Data
openapi_data = None

# Initialize Azure OpenAI Client
llm_client = AzureChatOpenAI(deployment_name="your-deployment-name", api_key="your-api-key", api_version="2023-03-15-preview")

import requests

def load_openapi_from_url_or_file(source: str):
    """Loads OpenAPI spec from a URL or file."""
    global openapi_data
    try:
        parser = OpenAPIParser()
        
        # Detect if source is a URL
        if source.startswith("http"):
            response = requests.get(source)
            if response.status_code == 200:
                openapi_data = parser.parse(response.json())  # Parse JSON response
            else:
                logging.error(f"Failed to fetch OpenAPI spec from URL. Status: {response.status_code}")
                return None
        else:
            openapi_data = parser.parse(source)  # Assume it's a file
        
        logging.info("✅ OpenAPI Spec Loaded Successfully")
        return openapi_data
    except Exception as e:
        logging.error(f"❌ Failed to load OpenAPI Spec: {e}")
        return None

# Mount static files directory (JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    """Serves the chat UI (index.html)."""
    return FileResponse("static/index.html")

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for API testing."""
    global openapi_data  # Ensure OpenAPI data is accessible
    await websocket.accept()
    await websocket.send_text("Welcome! Please provide the OpenAPI (Swagger) URL or upload a spec file.")

    llm = LLMSequenceGenerator(llm_client)  # Use AzureChatOpenAI
    
    while True:
        try:
            user_input = await websocket.receive_text()

            # Ensure OpenAPI spec is loaded before running any operations
            if openapi_data is None and "http" not in user_input:
                await websocket.send_text("No OpenAPI spec loaded. Please provide a Swagger URL first.")
                continue

            intent = llm.determine_intent(user_input, openapi_data)

            if intent == "provide_openapi":
                openapi_data = load_openapi_from_url_or_file(user_input)
                if openapi_data:
                    await websocket.send_text("✅ OpenAPI Spec Loaded! You can ask about available APIs or run tests.")
                else:
                    await websocket.send_text("❌ Failed to load OpenAPI Spec. Please check the URL and try again.")

            elif intent == "list_apis":
                if openapi_data:
                    apis = json.dumps(openapi_data.get_endpoints(), indent=2)
                    await websocket.send_text(f"📌 Available APIs:\n{apis}")
                else:
                    await websocket.send_text("⚠️ OpenAPI spec is not loaded. Please provide a URL first.")

            elif intent == "run_sequence":
                sequence = llm.suggest_sequence(openapi_data)
                await websocket.send_text(f"🔄 Suggested Execution Sequence: {sequence}. Confirm? (yes/no)")
                confirmation = await websocket.receive_text()
                if "yes" in confirmation.lower():
                    workflow_manager = APIWorkflowManager(openapi_data, llm_client)
                    result = await workflow_manager.execute_workflow(sequence)
                    await websocket.send_text(f"✅ Execution Results:\n{result}")

            elif intent == "load_test":
                num_users, duration = llm.extract_load_test_params(user_input)
                executor = APIExecutor()
                results = await executor.run_load_test(openapi_data, num_users, duration)
                await websocket.send_text(f"📊 Load Test Results:\n{results}")

            elif intent == "general_query":
                response = openapi_data.answer_query(user_input)
                await websocket.send_text(response)

            elif intent == "execute_api":
                method, endpoint = llm.extract_api_details(user_input)
                executor = APIExecutor()
                payload = llm.generate_payload(endpoint, openapi_data)
                result = await executor.execute_api(method, endpoint, payload)
                await websocket.send_text(f"🚀 Execution Result: {result}")

            elif intent == "modify_execution":
                await websocket.send_text("🔄 Would you like to modify the execution sequence? Provide new order.")
                new_sequence = await websocket.receive_text()
                workflow_manager = APIWorkflowManager(openapi_data, llm_client)
                modified_result = await workflow_manager.execute_workflow(json.loads(new_sequence))
                await websocket.send_text(f"✅ Modified Execution Results:\n{modified_result}")

            else:
                await websocket.send_text("🤔 I didn't understand. Try asking about APIs, running tests, or modifying the execution sequence.")
        
        except Exception as e:
            logging.error(f"Error in WebSocket communication: {e}")
            await websocket.send_text(f"❌ An error occurred: {str(e)}")
