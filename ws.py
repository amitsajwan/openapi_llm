from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
from typing import Dict, Any
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.runner import SubmitInterruptRequest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.schema import GraphContext
from api_executor import APIExecutor
from langgraph_workflow import LangGraphWorkflow, GraphState, GraphOutput

app = FastAPI()

@app.websocket("/chat")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # 1) Receive initial graph definition from client
        init_msg = await ws.receive_text()
        graph_def = GraphOutput.parse_raw(init_msg)

        # 2) Instantiate your API executor & workflow
        api_exec = APIExecutor(base_url="https://api.example.com")
        workflow = LangGraphWorkflow(
            graph_def=graph_def.dict(),
            api_executor=api_exec,
            websocket_callback=lambda t, payload: ws.send_json({"type": t, **payload})
        )

        # 3) Stream the graph execution
        #    runner.astream will yield either interrupt events or state snapshots
        async for event in workflow.runner.astream(GraphState()):
            # 4) Handle interrupt (human-in-the-loop)
            if isinstance(event, dict) and event.get("__type__") == "interrupt":
                intr = event  # contains keys: "__type__", "key", "name", "content"
                # send prompt to client
                await ws.send_json({
                    "type": "payload_confirmation",
                    "interrupt_key": intr["key"],
                    "prompt": intr["name"],
                    "payload": intr["content"]["payload"]
                })
                # wait for user’s edited payload
                resp = await ws.receive_text()
                msg = json.loads(resp)
                if msg.get("type") == "user_payload_confirmation":
                    # resume the workflow
                    await workflow.runner.submit_interrupt(
                        intr["key"],
                        msg["payload"]
                    )
                continue

            # 5) Normal node completion events include new operation_results
            if hasattr(event, "operation_results"):
                # find newly completed ops
                for op_id, result in event.operation_results.items():
                    await ws.send_json({
                        "type": "api_response",
                        "operationId": op_id,
                        "result": result
                    })

    except WebSocketDisconnect:
        # client disconnected
        pass
