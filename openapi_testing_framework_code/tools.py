import os
import json
import requests
from typing import Any, Dict
from models import BotState
from openapi_parser import OpenAPIParser

BASE_URL = os.getenv("OPENAPI_BASE_URL", "http://localhost:8000")

def load_spec(state: BotState, **kwargs) -> BotState:
    state.spec_path = state.plan[0].get("params", {}).get("spec_path") if state.plan else state.spec_path
    parser = OpenAPIParser(state.spec_path)
    state.openapi_schema = parser._raw_spec
    state.endpoints = [{"path": p, "method": m} for p in parser.get_all_paths() for m in parser.get_methods_for_path(p)]
    return state

def list_apis(state: BotState, **kwargs) -> BotState:
    state.text_content = {"endpoints": state.endpoints}
    return state

def generate_sequence(state: BotState, **kwargs) -> BotState:
    state.sequence = state.endpoints.copy()
    return state

def generate_payload(state: BotState, **kwargs) -> BotState:
    parser = OpenAPIParser(state.spec_path)
    for idx, step in enumerate(state.sequence):
        if step["method"].lower() in ["post", "put", "patch"]:
            schema = parser.get_request_schema(step["path"], step["method"])
            state.payloads[idx] = parser.generate_example_payload(schema or {})
    return state

def call_api(state: BotState, **kwargs) -> BotState:
    state.results = []
    for idx, step in enumerate(state.sequence):
        url = BASE_URL + step["path"]
        method = step["method"].lower()
        payload = state.payloads.get(idx)
        resp = requests.request(method, url, json=payload)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        state.results.append({"status": resp.status_code, "body": body})
    return state

def validate_graph(state: BotState, **kwargs) -> BotState:
    state.valid = all(r["status"] < 300 for r in state.results)
    return state

def save_results(state: BotState, **kwargs) -> BotState:
    with open("results.json", "w") as f:
        json.dump(state.results, f, indent=2)
    return state
