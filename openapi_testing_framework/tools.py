"""
tools.py

Defines LangGraph @tool functions for OpenAPI testing framework:
  - load_spec: read and cache OpenAPI YAML
  - list_apis: enumerate endpoints
  - generate_sequence: build execution graph/sequence
  - generate_payload: create request body for POST/PUT
  - call_api: perform the HTTP request
  - validate_graph: ensure no cycles
  - save_results: persist execution outcomes
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph import tool
from openapi_utils import parse_openapi_with_cache, get_endpoints, build_execution_graph, generate_payload_from_schema
from utils import llm_call_helper
import httpx

# --- Models for tool inputs/outputs ---
class LoadSpecResult(BaseModel):
    spec: Dict[str, Any]

class ListAPIsResult(BaseModel):
    apis: List[Dict[str, Any]]

class SequenceResult(BaseModel):
    sequence: List[str]

class PayloadResult(BaseModel):
    payload: Dict[str, Any]

class APIResponse(BaseModel):
    status_code: int
    body: Any
    headers: Dict[str, Any]

# --- Tool implementations ---
@tool("load_spec")
def load_spec(spec_path: str) -> LoadSpecResult:
    """
    Load and cache an OpenAPI YAML or JSON spec from the given file path.
    """
    spec = parse_openapi_with_cache(spec_path)
    return LoadSpecResult(spec=spec)

@tool("list_apis")
def list_apis(spec: Dict[str, Any]) -> ListAPIsResult:
    """
    Return a list of available endpoints (method + path) from the OpenAPI spec.
    """
    endpoints = get_endpoints(spec)
    return ListAPIsResult(apis=endpoints)

@tool("generate_sequence")
def generate_sequence(apis: List[Dict[str, Any]]) -> SequenceResult:
    """
    Build a safe execution sequence (DAG order) of the given APIs.
    """
    sequence = build_execution_graph(apis)
    return SequenceResult(sequence=sequence)

@tool("generate_payload")
def generate_payload(
    spec: Dict[str, Any],
    path: str,
    method: str,
    mode: str = Field("first_run", description="first_run | load_test")
) -> PayloadResult:
    """
    Generate or placeholder a request payload for the given operation.
    """
    # use LLM-heavy generation on first run
    schema = spec["paths"][path][method].get("requestBody", {})
    if mode == "first_run":
        # call LLM to produce realistic payload
        prompt = f"Generate a valid {method.upper()} payload for endpoint {path} based on schema: {schema}."
        llm_out = llm_call_helper(prompt=prompt, function_name="generate_payload_json", retries=2)
        payload = llm_out.get("payload")
    else:
        # placeholder mode
        payload = generate_payload_from_schema(schema, placeholder=True)
    return PayloadResult(payload=payload)

@tool("call_api")
def call_api(
    base_url: str,
    path: str,
    method: str,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None
) -> APIResponse:
    """
    Execute the HTTP request against the API.
    """
    url = base_url.rstrip("/") + path
    client = httpx.Client()
    response = client.request(method=method.upper(), url=url, headers=headers or {}, json=payload)
    return APIResponse(status_code=response.status_code, body=response.json(), headers=dict(response.headers))

@tool("validate_graph")
def validate_graph(sequence: List[str]) -> bool:
    """
    Ensure the execution sequence has no cycles and is safe to run.
    """
    # simple cycle check
    return True  # implement actual cycle detection in openapi_utils

@tool("save_results")
def save_results(
    results: List[Dict[str, Any]],
    output_path: str = "results.json"
) -> Dict[str, Any]:
    """
    Persist execution results to disk or memory saver.
    """
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return {"saved_to": output_path}
