Perfect — if you're planning to **call the LangGraph workflow using `astream` from outside**, you’ll want to expose an `astream` generator function that yields each state as the workflow progresses.

Here’s how you can modify the class to support that:

---

### ✅ Replace `.run()` with `.astream()`

Add this method to your existing `LanggraphWorkflow` class:

```python
    async def astream(self, initial_state: GraphState = None):
        """
        Asynchronously stream each state update in the graph.
        Yields state at each node, and allows human-in-the-loop payload confirmation.
        """
        state = initial_state or GraphState(base_url=self.base_url)

        async for step in self.graph.astream(state.model_dump()):
            current_state = GraphState(**step)

            # Human-in-the-loop check
            if current_state.first_run:
                for node_id, config in self.node_configs.items():
                    if config["method"].upper() in ["POST", "PUT"] and not current_state.confirmed.get(node_id):
                        payload = await self._get_payload_for_endpoint(node_id)
                        confirmed = await self.user_confirm_payload(node_id, payload)
                        current_state.payloads[node_id] = confirmed
                        current_state.confirmed[node_id] = True
                        current_state.first_run = False

                        # Restart from updated state
                        async for step in self.graph.astream(current_state.model_dump()):
                            yield GraphState(**step)
                        return

            yield current_state
```

---

### ✅ Example Usage with `async for`

```python
from langgraph_workflow import LanggraphWorkflow, GraphState
import asyncio

async def main():
    wf = LanggraphWorkflow(graph_json, base_url="http://localhost:8000")

    async for state in wf.astream():
        print("--- Step ---")
        print("Results:", state.results)

asyncio.run(main())
```

---

### What This Gives You

- You can **use it inside WebSockets or Streamlit async apps**.
- You can **intervene at any point**, especially on first `POST`/`PUT` for payload confirmation.
- You can **pause, inspect, modify state**, or just let it stream through.

Let me know if you want an `observe_stream` wrapper or to plug this directly into a chat or UI system.
