import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langgraph import tool

from utils import llm_call_helper

class RouterOutput(BaseModel):
    plan: List[Dict[str, Any]] = Field(..., description="Sequence of actions to execute.")
    next_action: str = Field(..., description="Tool function name for the next step.")
    next_params: Dict[str, Any] = Field(default_factory=dict)

@tool("router")
def router(state: Any, user_input: str) -> RouterOutput:
    plan = getattr(state, "plan", None)
    if not plan:
        prompt = f"Create a plan of actions for: {user_input}"
        resp = llm_call_helper(prompt, function_name="create_plan")
        plan = resp.get("plan", [])
        state.plan = plan
    next_step = state.plan.pop(0)
    action = next_step.get("action")
    params = next_step.get("params", {})
    if any(v is None for v in params.values()):
        prompt = f"Extract params for {action} from: {user_input}"
        p = llm_call_helper(prompt, function_name="extract_params")
        params.update(p)
    return RouterOutput(plan=state.plan, next_action=action, next_params=params)
