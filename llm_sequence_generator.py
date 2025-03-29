import json
from langchain_openai import AzureChatOpenAI

class LLMSequenceGenerator:
    def __init__(self, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client

    def determine_intent(self, user_input, openapi_data):
        """Determines the intent of the user query based on OpenAPI data."""
        prompt = (
            f"User Query: {user_input}\n"
            "Analyze the query based on OpenAPI schema and determine the intent: "
            "provide_openapi, list_apis, run_sequence, load_test, execute_api, general_query, or modify_execution."
        )
        response = self._ask_llm(prompt)
        return response.lower().strip()

    def suggest_sequence(self, openapi_data):
        """Suggests an execution sequence based on the OpenAPI specification."""
        prompt = (
            "Analyze the OpenAPI schema and determine the optimal execution order for API endpoints. "
            "Return the ordered list of API calls with HTTP methods and required parameters in JSON format."
        )
        response = self._ask_llm(prompt)
        return json.loads(response)

    def extract_load_test_params(self, user_input):
        """Extracts the number of users and duration for a load test from user input."""
        prompt = (
            f"Extract load test parameters (number of users, duration in seconds) from: '{user_input}'. "
            "Return a JSON object with 'num_users' and 'duration'."
        )
        response = self._ask_llm(prompt)
        params = json.loads(response)
        return params.get("num_users", 1), params.get("duration", 60)

    def extract_api_details(self, user_input):
        """Extracts API method and endpoint details from user input."""
        prompt = (
            f"Extract the API method (GET, POST, etc.) and endpoint from: '{user_input}'. "
            "Return a JSON object with 'method' and 'endpoint'."
        )
        response = self._ask_llm(prompt)
        details = json.loads(response)
        return details.get("method"), details.get("endpoint")

    def generate_payload(self, endpoint, openapi_data):
        """Generates a sample payload based on OpenAPI schema for a given endpoint."""
        prompt = (
            f"Generate a sample JSON payload for the API endpoint '{endpoint}' based on OpenAPI schema."
        )
        response = self._ask_llm(prompt)
        return json.loads(response)

    def _ask_llm(self, prompt):
        """Interacts with Azure OpenAI Chat API to generate responses."""
        messages = [
            {"role": "system", "content": "You are an API assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client(messages=messages)
        return response['choices'][0]['message']['content']
