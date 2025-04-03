import json
import asyncio
import websockets

async def websocket_handler(websocket, path):
    while True:
        # Example Mermaid graph
        mermaid_diagram = """
        graph TD;
            A[Start] --> B[Process]
            B --> C[End]
        """

        # Create JSON message
        message = json.dumps({
            "type": "mermaid_graph",
            "data": mermaid_diagram
        })

        # Send JSON to WebSocket client
        await websocket.send(message)
        await asyncio.sleep(5)  # Send updates every 5 seconds (example)

# Start WebSocket server
start_server = websockets.serve(websocket_handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
