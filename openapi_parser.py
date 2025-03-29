import yaml

class OpenAPIParser:
    def __init__(self):
        self.openapi_spec = None

    def parse(self, source: str):
        """Parses the OpenAPI YAML file from a URL or local path."""
        if source.startswith("http"):
            import requests
            response = requests.get(source)
            self.openapi_spec = yaml.safe_load(response.text)
        else:
            with open(source, "r") as file:
                self.openapi_spec = yaml.safe_load(file)
        return self.openapi_spec

    def get_endpoints(self):
        """Extracts API endpoints from the OpenAPI spec."""
        return list(self.openapi_spec.get("paths", {}).keys())

    def answer_query(self, query: str):
        """Answers general queries based on OpenAPI spec."""
        if "authentication" in query.lower():
            return self.openapi_spec.get("components", {}).get("securitySchemes", "No authentication details found.")
        elif "server" in query.lower():
            return self.openapi_spec.get("servers", "No server details found.")
        else:
            return "I can answer questions about authentication, servers, and endpoints."
