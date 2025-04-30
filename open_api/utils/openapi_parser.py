import yaml
from typing import Any, Dict

class OpenAPIParser:
    """
    Parses an OpenAPI (v3) specification from YAML or JSON text into a Python dict.
    """
    def parse_from_text(self, spec_text: str) -> Dict[str, Any]:
        """
        Load the OpenAPI spec into a dictionary.
        Supports both JSON and YAML formats.
        """
        try:
            # PyYAML safe load handles both JSON and YAML :contentReference[oaicite:4]{index=4}
            spec = yaml.safe_load(spec_text)
            return spec
        except Exception as e:
            raise ValueError(f"Failed to parse OpenAPI spec: {e}")
