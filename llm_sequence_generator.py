import json
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import AzureChatOpenAI

class LLMSequenceGenerator:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.runnable = RunnablePassthrough()

    async def determine_intent(self, user_input: str, openapi_data: dict = None):
        """Determines user intent based on OpenAPI specification."""
        prompt = (
            f"Identify intent for user query based on OpenAPI schema.\n"
            f"Query: {user_input}\n"
            f"Schema: {json.dumps(openapi_data, indent=2) if openapi_data else 'None'}\n"
            f"Return only intent as JSON: {{'intent': 'value'}}"
        )
        response = await self.llm.apredict(prompt)
        return json.loads(response).get("intent", "unknown")

    async def suggest_sequence(self, openapi_data: dict):
        """Suggests execution sequence for APIs based on dependencies."""
        prompt = (
            f"Analyze the OpenAPI schema and suggest execution order.\n"
            f"Schema: {json.dumps(openapi_data, indent=2)}\n"
            f"Return a JSON list of API execution steps."
        )
        response = await self.llm.apredict(prompt)
        return json.loads(response)

    async def extract_load_test_params(self, user_input: str):
        """Extracts load test parameters from user input."""
        prompt = (
            f"Extract load test parameters (num_users, duration) from query: {user_input}\n"
            f"Return JSON: {{'num_users': int, 'duration': int}}"
        )
        response = await self.llm.apredict(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"num_users": None, "duration": None}

    async def extract_api_details(self, user_input: str):
        """Extracts API method and endpoint from user input."""
        prompt = (
            f"Identify API method and endpoint from query: {user_input}\n"
            f"Return JSON: {{'method': 'GET/POST/etc', 'endpoint': '/example'}}"
        )
        response = await self.llm.apredict(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"method": None, "endpoint": None}

    async def generate_payload(self, endpoint: str, openapi_data: dict):
        """Generates a payload for a given API endpoint."""
        prompt = (
            f"Generate a realistic JSON payload for endpoint {endpoint} using OpenAPI schema.\n"
            f"Schema: {json.dumps(openapi_data, indent=2)}\n"
            f"Return only JSON."
        )
        response = await self.llm.apredict(prompt)
        return json.loads(response)

    async def answer_general_query(self, user_input: str):
        """Answers general queries that do not match specific intents."""
        prompt = f"Answer this general query concisely: {user_input}"
        response = await self.llm.apredict(prompt)
        return response.strip()
        
