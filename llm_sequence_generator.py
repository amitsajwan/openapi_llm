import json
from langchain_openai import AzureChatOpenAI

class LLMSequenceGenerator:
    def __init__(self, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client

    async def determine_intent(self, user_input: str, openapi_data):
        """Determines user intent based on input and OpenAPI context."""
        prompt = f"User input: {user_input}\nOpenAPI Context: {json.dumps(openapi_data, indent=2) if openapi_data else 'No OpenAPI Data'}\nIntent:"
        response = await self.llm_client.ainvoke(prompt)
        return response.content.strip()

    async def suggest_sequence(self, openapi_data):
        """Suggests an API execution sequence."""
        prompt = f"Given the following OpenAPI data, suggest a sequence of API calls:\n{json.dumps(openapi_data, indent=2)}"
        response = await self.llm_client.ainvoke(prompt)
        return json.loads(response.content)

    async def answer_general_query(self, user_input: str, openapi_data):
        """Answers general user queries based on OpenAPI data."""
        prompt = f"User query: {user_input}\nOpenAPI Context: {json.dumps(openapi_data, indent=2)}\nAnswer:"
        response = await self.llm_client.ainvoke(prompt)
        return response.content.strip()

    async def extract_load_test_params(self, user_input: str):
        """Extracts number of users and duration for load testing."""
        prompt = f"Extract the number of users and duration from: {user_input}"
        response = await self.llm_client.ainvoke(prompt)
        params = json.loads(response.content)
        return params.get("num_users", 1), params.get("duration", 60)

    async def extract_api_details(self, user_input: str):
        """Extracts API method and endpoint from user input."""
        prompt = f"Extract API method and endpoint from: {user_input}"
        response = await self.llm_client.ainvoke(prompt)
        details = json.loads(response.content)
        return details.get("method"), details.get("endpoint")

    async def generate_payload(self, endpoint: str, openapi_data):
        """Generates a payload based on the OpenAPI specification."""
        prompt = f"Generate an example payload for the endpoint {endpoint} based on the following OpenAPI data:\n{json.dumps(openapi_data, indent=2)}"
        response = await self.llm_client.ainvoke(prompt)
        return json.loads(response.content)
