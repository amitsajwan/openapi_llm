"""
router.py

Defines the LangGraph router node which interprets user intents, generates multi-step plans,
extracts parameters, and dispatches execution to tool functions.

We use an LLM-heavy approach: the router will call the LLM to (1) decide on the next intent or plan,
(2) extract any required parameters for the next action, and (3) output a structured JSON dispatch.

The router integrates Solution 2's robust LLM calls (with retries) and Solution 1's multi-step plan concept.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph import tool, StateGraph
from utils import llm_call_helper


# Pydantic model for router's output
class RouterOutput(BaseModel):
    # A high-level plan of steps, each step has an "action" and optional "params"
    plan: List[Dict[str, Any]] = Field(..., description="Sequence of actions to execute.")
    # The next immediate action to run (popped from plan)
    next_action: str = Field(..., description="Tool function name for the next step.")
    # Parameters for the next action
    next_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the next tool.")


@tool("router")
def router(state: BaseModel, user_input: str) -> RouterOutput:
    """
    LangGraph router node:
     1. If there is no existing plan in state, ask the LLM to generate a multi-step plan.
     2. Extract parameters for the next action via LLM.
     3. Return RouterOutput with plan, next_action, next_params.

    State is expected to have an attribute `plan` (List[Dict]) in its scratchpad or bot state.
    """
    # 1. Check if a plan already exists in state
    plan: Optional[List[Dict[str, Any]]] = getattr(state, 'plan', None)

    if not plan:
        # Generate a new multi-step plan
        plan_prompt = f"""
You are an API testing assistant. Given the user request: "{user_input}",
create a step-by-step plan. Represent the plan as a JSON list of {{action: <tool_name>, params: {{...}}}}."""
        plan_response = llm_call_helper(
            prompt=plan_prompt,
            function_name="generate_plan",
            retries=3
        )
        # parse JSON
        plan = plan_response.get('plan')
        # store the plan back into state
        state.plan = plan

    # 2. Pop the next step
    next_step = state.plan.pop(0)
    action = next_step['action']
    params = next_step.get('params', {})

    # 3. If any params are missing or need extraction, ask LLM to extract
    if any(v is None for v in params.values()):
        param_prompt = f"""
Extract required parameters for action '{action}' from user input: "{user_input}".
Return a JSON object matching the tool signature."""
        param_response = llm_call_helper(
            prompt=param_prompt,
            function_name=f"extract_params_for_{action}",
            retries=2
        )
        params.update(param_response)

    return RouterOutput(
        plan=state.plan,
        next_action=action,
        next_params=params
    )
