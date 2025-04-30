from typing import Any, Dict
from jsonschema import Draft7Validator

class PayloadGenerator:
    """
    Generates example payloads for each request body schema in the OpenAPI spec.
    """
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec

    def _generate_example(self, schema: Dict[str, Any]) -> Any:
        """
        Recursively generate example data for a JSON schema node.
        """
        t = schema.get("type")
        if t == "object":
            example = {}
            for prop, subschema in schema.get("properties", {}).items():
                example[prop] = self._generate_example(subschema)
            return example
        if t == "array":
            return [self._generate_example(schema.get("items", {}))]
        if t == "string":
            return schema.get("example", "string")  # use provided example if any :contentReference[oaicite:7]{index=7}
        if t in ("integer", "number"):
            return schema.get("example", 0)
        if t == "boolean":
            return schema.get("example", True)
        return None

    def generate_all(self) -> Dict[str, Any]:
        """
        For each path + method with a requestBody, generate a sample payload.
        """
        payloads: Dict[str, Any] = {}
        for path, methods in self.spec.get("paths", {}).items():
            for method, details in methods.items():
                op_id = details.get("operationId", f"{method}_{path}")
                rb = details.get("requestBody", {}).get("content", {}).get("application/json", {})
                schema = rb.get("schema")
                if schema:
                    example = self._generate_example(schema)
                    payloads[op_id] = example
        return payloads
