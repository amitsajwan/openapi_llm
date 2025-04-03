from fastapi import FastAPI, WebSocket
import asyncio
from langgraph.graph import StateGraph

app = FastAPI()

# WebSocket clients
clients = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    
    try:
        while True:
            await asyncio.sleep(5)  # Periodically send updates
            mermaid_code = get_mermaid_code()  # Function to generate Mermaid.js diagram
            for client in clients:
                await client.send_text(mermaid_code)
    except Exception:
        clients.remove(websocket)

def get_mermaid_code():
    """Generate the latest Mermaid diagram from LangGraph"""
    return graph.get_graph().draw_mermaid()  # Replace `graph` with your actual instance
