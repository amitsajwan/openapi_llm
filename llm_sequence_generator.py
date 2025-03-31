import json
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI

class LLMSequenceGenerator:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.intent_runnable = RunnablePassthrough()
        self.json_runnable = RunnablePassthrough()
        self.text_runnable = RunnablePassthrough()
    
    async def determine_intent(self, user_input: str, openapi_data: dict):
        """Determines user intent and ensures a valid response."""
        prompt = f"Determine the intent of the following user input based on OpenAPI schema: {user_input}.\nReturn a single-word intent."
        response = await self.intent_runnable.invoke(self.llm.apredict(prompt))
        return response.strip()
    
    async def suggest_sequence(self, openapi_data: dict):
        """Suggests execution order of APIs in OpenAPI spec."""
        prompt = f"Suggest an execution sequence for APIs in this OpenAPI schema: {json.dumps(openapi_data)}. Return JSON."
        response = await self.json_runnable.invoke(self.llm.apredict(prompt))
        return json.loads(response)
    
    async def extract_load_test_params(self, user_input: str):
        """Extracts load test parameters from input."""
        prompt = f"Extract load test parameters (num_users, duration) from: {user_input}. Return JSON."
        response = await self.json_runnable.invoke(self.llm.apredict(prompt))
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    
    async def extract_api_details(self, user_input: str):
        """Extracts API method and endpoint."""
        prompt = f"Extract API details (method, endpoint) from: {user_input}. Return JSON."
        response = await self.json_runnable.invoke(self.llm.apredict(prompt))
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    
    async def generate_payload(self, endpoint: str, openapi_data: dict):
        """Generates a valid JSON payload for an API."""
        prompt = f"Generate a valid JSON payload for endpoint {endpoint} using OpenAPI schema: {json.dumps(openapi_data)}. Return JSON."
        response = await self.json_runnable.invoke(self.llm.apredict(prompt))
        return json.loads(response)
    
    async def answer_general_query(self, user_input: str):
        """Handles general queries and ensures proper text formatting."""
        prompt = f"Answer the following general query in a well-formatted manner: {user_input}"
        response = await self.text_runnable.invoke(self.llm.apredict(prompt))
        return response.strip()
