from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
import json
from openai import OpenAI
from openapi_parser import OpenAPIParser
from api_workflow import APIWorkflowManager
from api_executor import APIExecutor
from llm_sequence_generator import LLMSequenceGenerator
from utils.result_storage import ResultStorage

app = FastAPI()
openai_client = OpenAI(api_key="your-api-key")

# Global Initializations
openapi_file = "openapi_specs/petstore.yaml"
base_url = "https://petstore.swagger.io/v2"
auth_headers = {}

parser = OpenAPIParser(openapi_file)
api_map = parser.get_all_endpoints()
llm_gen = LLMSequenceGenerator()
api_executor = APIExecutor(base_url, auth_headers)
workflow_manager = APIWorkflowManager(base_url, auth_headers)

connected_clients = set()

@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        await websocket.send_text("👋 Welcome! Ask about the API or say 'run tests' to begin.")

        while True:
            user_input = await websocket.receive_text()
            response = await process_user_input(user_input)
            await websocket.send_text(response)

    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def process_user_input(user_input):
    """Processes user commands via OpenAI and manages API execution."""
    llm_response = openai_client.chat.create(model="gpt-4", messages=[{"role": "user", "content": user_input}])
    intent = llm_response["choices"][0]["message"]["content"]

    if "run tests" in intent:
        sequence = llm_gen.generate_sequence(api_map)
        return f"Suggested execution order: {sequence}. Confirm? (yes/no)"
    elif "yes" in intent:
        return await workflow_manager.execute_workflow(sequence)
    elif "load test" in intent:
        return await api_executor.run_load_test(sequence)
    
    return f"🤖 AI Response: {intent}"

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
