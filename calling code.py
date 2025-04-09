from fastapi import FastAPI, WebSocket
from langgraph_workflow import LanggraphWorkflow, llm_refiner
import asyncio

app = FastAPI()

# Dummy graph definition
sample_graph = {
    "nodes": [
        {"id": "createProduct", "method": "post", "endpoint": "/product", "operation_id": "createProduct"},
        {"id": "getProduct", "method": "get", "endpoint": "/product/{id}", "operation_id": "getProduct"}
    ],
    "edges": [
        {"source": "createProduct", "target": "getProduct"}
    ]
}

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()

    workflow = LanggraphWorkflow(
        graph_def=sample_graph,
        websocket=websocket,
        llm_refiner=llm_refiner
    )

    workflow.build_graph()

    await websocket.send_json({
        "type": "workflow_start",
        "message": "Workflow started. Executing now..."
    })

    await workflow.astream()

    await websocket.send_json({
        "type": "workflow_complete",
        "message": "Workflow execution completed."
    })
