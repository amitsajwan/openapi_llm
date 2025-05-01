import os
from typing import Dict, List, Optional, Tuple

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
        self._reduced_spec = ReducedOpenAPISpec.from_spec_dict(self._spec.raw_spec)
        self._toolkit = None

        if llm:
            self._toolkit = OpenAPIToolkit.from_llm_and_reduced_spec(llm, self._reduced_spec)

    def get_all_paths(self) -> List[str]:
        """Returns a list of all available paths from the OpenAPI spec."""
        return list(self._spec.raw_spec.get("paths", {}).keys())

    def get_methods_for_path(self, path: str) -> List[str]:
        """Returns available HTTP methods for a given path."""
        return list(self._spec.raw_spec.get("paths", {}).get(path, {}).keys())

    def get_request_schema(self, path: str, method: str) -> Optional[Dict]:
        """Returns the request body schema for a given endpoint and method."""
        method = method.lower()
        method_obj = self._spec.raw_spec["paths"].get(path, {}).get(method)
        if not method_obj:
            return None

        request_body = method_obj.get("requestBody")
        if not request_body:
            return None

        # Handle content type
        content = request_body.get("content", {})
        if "application/json" in content:
            return content["application/json"].get("schema")
        elif content:
            # Fallback to any content type
            return next(iter(content.values())).get("schema")
        return None

    def get_response_schema(self, path: str, method: str, status_code: str = "200") -> Optional[Dict]:
        """Returns the response schema for a given endpoint and method."""
        method = method.lower()
        method_obj = self._spec.raw_spec["paths"].get(path, {}).get(method)
        if not method_obj:
            return None

        responses = method_obj.get("responses", {})
        response = responses.get(status_code)
        if not response:
            return None

        content = response.get("content", {})
        if "application/json" in content:
            return content["application/json"].get("schema")
        elif content:
            return next(iter(content.values())).get("schema")
        return None

    def get_summary(self, path: str, method: str) -> str:
        """Returns the summary or operationId for display."""
        method = method.lower()
        method_obj = self._spec.raw_spec["paths"].get(path, {}).get(method, {})
        return method_obj.get("summary") or method_obj.get("operationId", "")

    def get_toolkit(self) -> Optional[OpenAPIToolkit]:
        """Returns the LangChain OpenAPIToolkit (only available if LLM is passed)."""
        return self._toolkit

    def get_tools(self):
        """Returns LangChain-compatible tools for LangGraph (LLM required)."""
        if not self._toolkit:
            raise ValueError("LLM not provided. Toolkit not initialized.")
        return self._toolkit.get_tools()
