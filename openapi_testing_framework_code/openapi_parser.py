import os
from typing import Dict, List, Optional, Union, Any

from langchain_community.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain_community.tools.openapi.utils.openapi_parser import ReducedOpenAPISpec
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_core.language_models.chat_models import BaseChatModel

class OpenAPIParser:
    def __init__(self, yaml_path: str, llm: Optional[BaseChatModel] = None):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Spec file not found: {yaml_path}")
        self.yaml_path = yaml_path
        self.llm = llm
        self._spec = OpenAPISpec.from_file(yaml_path)
        self._raw_spec = self._spec.raw_spec
        self._components = self._raw_spec.get("components", {}).get("schemas", {})
        self._reduced_spec = ReducedOpenAPISpec.from_spec_dict(self._raw_spec)
        self._toolkit = OpenAPIToolkit.from_llm_and_reduced_spec(llm, self._reduced_spec) if llm else None

    def get_all_paths(self) -> List[str]:
        return list(self._raw_spec.get("paths", {}).keys())

    def get_methods_for_path(self, path: str) -> List[str]:
        return list(self._raw_spec.get("paths", {}).get(path, {}).keys())

    def get_summary(self, path: str, method: str) -> str:
        method_obj = self._raw_spec["paths"].get(path, {}).get(method.lower(), {})
        return method_obj.get("summary") or method_obj.get("operationId", "")

    def get_request_schema(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        method_obj = self._raw_spec["paths"].get(path, {}).get(method.lower())
        if not method_obj:
            return None
        request_body = method_obj.get("requestBody", {}).get("content", {})
        schema = None
        if "application/json" in request_body:
            schema = request_body["application/json"].get("schema")
        elif request_body:
            schema = next(iter(request_body.values())).get("schema")
        return self.resolve_schema(schema) if schema else None

    def get_response_schema(self, path: str, method: str, status_code: str = "200") -> Optional[Dict[str, Any]]:
        method_obj = self._raw_spec["paths"].get(path, {}).get(method.lower())
        if not method_obj:
            return None
        responses = method_obj.get("responses", {})
        resp = responses.get(status_code)
        if not resp:
            return None
        content = resp.get("content", {})
        schema = None
        if "application/json" in content:
            schema = content["application/json"].get("schema")
        elif content:
            schema = next(iter(content.values())).get("schema")
        return self.resolve_schema(schema) if schema else None

    def resolve_schema(self, schema: Optional[Union[Dict, str]]) -> Optional[Dict[str, Any]]:
        if not schema:
            return None
        if "$ref" in schema:
            ref = schema["$ref"].split("/")[-1]
            return self.resolve_schema(self._components.get(ref))
        if "allOf" in schema:
            out = {}
            for s in schema["allOf"]:
                part = self.resolve_schema(s)
                if part:
                    out.update(part)
            return out
        if "anyOf" in schema or "oneOf" in schema:
            key = "anyOf" if "anyOf" in schema else "oneOf"
            return self.resolve_schema(schema[key][0])
        return schema

    def generate_example_payload(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not schema:
            return {}
        example: Dict[str, Any] = {}
        props = schema.get("properties", {})
        for prop, ps in props.items():
            ps = self.resolve_schema(ps)
            t = ps.get("type", "string")
            if "enum" in ps:
                example[prop] = ps["enum"][0]
            elif t == "string":
                example[prop] = "string"
            elif t == "integer":
                example[prop] = 0
            elif t == "number":
                example[prop] = 0.0
            elif t == "boolean":
                example[prop] = True
            elif t == "array":
                items = self.resolve_schema(ps.get("items", {}))
                example[prop] = [self.generate_example_payload(items) if isinstance(items, dict) else None]
            elif t == "object":
                example[prop] = self.generate_example_payload(ps)
            else:
                example[prop] = None
        return example

    def get_toolkit(self) -> Optional[OpenAPIToolkit]:
        return self._toolkit

    def get_tools(self):
        if not self._toolkit:
            raise ValueError("LLM not provided. Toolkit not initialized.")
        return self._toolkit.get_tools()
