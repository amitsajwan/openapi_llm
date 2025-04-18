from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import asyncio
from langgraph_workflow import LangGraphWorkflow, GraphState
from api_executor import APIExecutor
from intent_router import intent_router

app = FastAPI()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    workflow = None  # To persist between commands

    async def send_callback(msg_type: str, payload: dict):
        await websocket.send_json({"type": msg_type, **payload})

    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)

            # Step 1: Let the bot process the user message
            response = await intent_router.handle_user_input(data_json["message"])

            if response.get("intent") == "api_execution_graph":
                # Graph creation intent
                content_dict = json.loads(response["response"])
                graph_data = content_dict["graph"]
                text_content = content_dict["textContent"]

                # Build LangGraph workflow and store it
                workflow = LangGraphWorkflow(
                    graph_def=graph_data,
                    api_executor=APIExecutor(base_url="http://localhost:8000"),
                    websocket_callback=send_callback
                )

                await websocket.send_json({"type": "graph", "sender": "bot", "graph": graph_data})
                await websocket.send_json({"type": "bot", "sender": "bot", "message": text_content})

            elif response.get("intent") == "execute_workflow":
                if workflow is None:
                    await websocket.send_json({
                        "type": "bot",
                        "sender": "bot",
                        "message": "No workflow is defined yet. Please generate the graph first."
                    })
                    continue

                await websocket.send_json({
                    "type": "workflow_start",
                    "sender": "bot",
                    "message": "Workflow started. Executing now..."
                })

                # Run workflow
                initial_state = GraphState()

                async for _ in workflow.astream(initial_state):
                    pass  # All messaging handled internally

                await websocket.send_json({
                    "type": "workflow_complete",
                    "sender": "bot",
                    "message": "Workflow execution completed."
                })

            elif response.get("intent") == "user_payload_confirmation":
                key = response["response"]["interrupt_key"]
                payload = response["response"]["payload"]
                if workflow:
                    await workflow.submit_interrupt_response(key, payload)

            else:
                await websocket.send_json({
                    "sender": "bot",
                    "type": "bot",
                    "message": response["response"]
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        await websocket.close()
