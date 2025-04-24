state = initial_state or GraphState()
state_dict = state.dict()
cfg = config or {}

# Validate required checkpoint config
if "thread_id" not in cfg:
    raise ValueError("Missing thread_id in config for checkpointing")

# Start astream only once
gen = self.runner.astream(state_dict, cfg, stream_mode="updates")

while True:
    try:
        step = await gen.__anext__()  # Get next update

        # Handle human-in-the-loop interrupt
        if "loop_interrupt" in step:
            interrupt = step["loop_interrupt"]

            if interrupt.get("type") in ("payload_confirmation", "question"):
                op_id = interrupt["value"].get("operation_id", "")
                path = interrupt["value"].get("path", "")
                method = interrupt["value"].get("method", "").upper()

                await self.websocket_callback(
                    "payload_confirmation",
                    {
                        "question": interrupt["value"].get("question", ""),
                        "operation_id": op_id,
                        "path": path,
                        "method": method,
                        "payload": interrupt["value"].get("payload", {}),
                    },
                )

                try:
                    resume_value = await asyncio.wait_for(
                        self._resume_queue.get(), timeout=60
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError("User did not resume in time")

                await gen.asend(resume_value)  # Resume the same generator

                continue  # Continue to next loop after resume

        # Process step outputs
        for val in step.get("items", []):
            for op in val.get("operations", []):
                status = op.get("result", {}).get("status_code", 0)
                success = status < 300
                exec_time = op.get("result", {}).get("execution_time", 0)

                msg = f"{op.get('method')} {op.get('path')} -> {status} [{exec_time}ms]"

                await self.websocket_callback("api_response", {"message": msg})

                if not success:
                    await self.websocket_callback(
                        "api_error",
                        {"message": "API call failed. Check logs."},
                    )
                    return

    except StopAsyncIteration:
        await self.websocket_callback("workflow_complete", {"message": "Execution complete."})
        break
