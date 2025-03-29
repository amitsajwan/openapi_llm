import yaml
import requests

class OpenAPIParser:
    def parse(self, source):
        if source.startswith("http"):
            response = requests.get(source)
            spec = yaml.safe_load(response.text)
        else:
            with open(source, "r") as file:
                spec = yaml.safe_load(file)

        return OpenAPISpec(spec)

class OpenAPISpec:
    def __init__(self, spec):
        self.spec = spec

    def get_endpoints(self):
        return list(self.spec["paths"].keys())

    def answer_query(self, query):
        if "authentication" in query.lower():
            return self.spec.get("security", "No authentication details found.")
        if "parameters" in query.lower():
            return {path: details.get("parameters", []) for path, details in self.spec["paths"].items()}
        return "I can help with API details, execution, or load tests."
