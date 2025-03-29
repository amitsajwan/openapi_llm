import yaml

class OpenAPIParser:
    def __init__(self, openapi_file):
        with open(openapi_file, "r") as file:
            self.openapi_spec = yaml.safe_load(file)

    def get_all_endpoints(self):
        return [f"{method.upper()} {path}" for path, methods in self.openapi_spec["paths"].items() for method in methods]
