"""
models.py

Pydantic models for shared state and data exchange in the OpenAPI testing framework.

- BotState: overall conversation/workflow state
- GraphOutput: captures the router output for LangGraph
- ToolParam models: placeholders for function signatures
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class BotState(BaseModel):
    """
    Represents the current state of the chatbot/workflow.

    Attributes:
      plan: remaining sequence of steps (action + params)
      spec_path: path to the selected OpenAPI spec
      spec: parsed OpenAPI spec dict
      apis: extracted list of endpoints
      sequence: planned execution order of endpoint keys
      results: list of API call results
      base_url: target API base URL
      headers: optional HTTP headers
      placeholders: stored placeholder IDs or values for reuse
    """
    plan: Optional[List[Dict[str, Any]]] = Field(default=None)
    spec_path: Optional[str] = Field(default=None)
    spec: Optional[Dict[str, Any]] = Field(default=None)
    apis: Optional[List[Dict[str, Any]]] = Field(default=None)
    sequence: Optional[List[str]] = Field(default=None)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    base_url: Optional[str] = Field(default=None)
    headers: Optional[Dict[str, str]] = Field(default=None)
    placeholders: Dict[str, Any] = Field(default_factory=dict)


class GraphOutput(BaseModel):
    """
    Standard structure for router output consumed by LangGraph edges.
    """
    plan: List[Dict[str, Any]]
    next_action: str
    next_params: Dict[str, Any]


# Tool parameter models (for documentation / type-hinting)
class LoadSpecParams(BaseModel):
    spec_path: str

class ListAPIsParams(BaseModel):
    spec: Dict[str, Any]

class GenerateSequenceParams(BaseModel):
    apis: List[Dict[str, Any]]

class GeneratePayloadParams(BaseModel):
    spec: Dict[str, Any]
    path: str
    method: str
    mode: Optional[str] = "first_run"

class CallAPIParams(BaseModel):
    base_url: str
    path: str
    method: str
    headers: Optional[Dict[str, str]]
    payload: Optional[Dict[str, Any]]

class SaveResultsParams(BaseModel):
    results: List[Dict[str, Any]]
    output_path: Optional[str] = "results.json"
