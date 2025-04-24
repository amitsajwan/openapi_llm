from typing import Optional, Dict, Any
import asyncio
from langgraph.types import Command, interrupt
from langgraph.graph import GraphState

async def astream(self,
                  initial_state: Optional[GraphState] = None,
                  config: Optional[Dict[str, Any]] = None):
    """
    Stream the graph execution, handle multiple human-in-the-loop interrupts,
    and ensure progression to subsequent nodes by updating state on each resume.
    """
    # 1. Prepare initial state and config
    state = (initial_state or GraphState())
    state_dict = state.dict()
    cfg = config or {}
    if "thread_id" not in cfg:
        raise ValueError("Missing thread_id in config for checkpointing")

    # 2. Start streaming execution
    gen = self.runner.astream(state_dict, cfg, stream_mode="updates")

    while True:
        try:
            async for mode, step in gen:
                # 3. Handle human-in-the-loop interrupt
                if mode == "updates" and "_interrupt_" in step:
                    intr = step["_interrupt_"][0]
                    # Send payload for confirmation
                    prompt = intr.value.get("question") or (
                        f"Operation: {intr.value['operationId']} "
                        f"Method: {intr.value['method']} "
                        f"Path: {intr.value['path']} – confirm payload")
                    await self.websocket_callback("payload_confirmation", {
                        "interrupt_key": intr.ns,
                        "prompt": prompt,
                        "payload": intr.value.get("payload", {}),
                    })  

                    # 4. Wait for user to resume with a value
                    try:
                        resume_value = await asyncio.wait_for(
                            self._resume_queue.get(), timeout=60
                        )
                    except asyncio.TimeoutError:
                        raise RuntimeError("User did not resume in time")

                    # 5. Update graph state with the resume value
                    #    Keyed by the interrupt namespace so next .astream moves on
                    state_dict[intr.ns] = resume_value
                    gen = self.runner.astream(state_dict, cfg, stream_mode="updates")  # reset generator
                    break  # restart async-for on updated state

                # 6. Process normal step outputs
                for _, value in step.items():
                    for op in value["operations"]:
                        status = op["result"].get("status_code")
                        success = status == 200
                        exec_time = op["result"].get("execution_time")
                        # record metrics
                        self.api_latency_seconds.labels(
                            name=op["result"].get("api", "")
                        ).observe(exec_time)  :contentReference[oaicite:1]{index=1}

                        # send user-friendly update
                        msg = (
                            f"Operation: {op['operationId']} "
                            f"Status: {'✅ Success' if success else '❌ Failure'}\n"
                            f"Path: {op['result'].get('path','')}  "
                            f"Code: {status}  "
                            f"Time: {exec_time}s"
                        )
                        await self.websocket_callback("api_response", {"message": msg})
                        if not success:
                            await self.websocket_callback(
                                "workflow_complete",
                                {"message": "API call failed. Check logs."}
                            )
                            return
        except StopAsyncIteration:
            # 7. Completed all nodes
            await self.websocket_callback("workflow_complete", {
                "message": "Workflow execution completed."
            })
            break
            
