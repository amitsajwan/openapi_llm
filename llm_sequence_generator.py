from openai import AzureChatOpenAI

class LLMSequenceGenerator:
    def __init__(self, llm_client: AzureChatOpenAI):
        """Initialize with an AzureChatOpenAI client."""
        self.llm_client = llm_client

    async def determine_intent(self, user_input, openapi_data):
        """Use AzureChatOpenAI to determine user intent."""
        prompt = f"User asked: {user_input}\nBased on OpenAPI data, classify the intent as: provide_openapi, list_apis, run_sequence, load_test, general_query, execute_api, modify_execution."
        
        response = await self.llm_client.acompletion_create(
            messages=[{"role": "system", "content": "You are an API assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        return response["choices"][0]["message"]["content"].strip()

    async def suggest_sequence(self, openapi_data):
        """Use AzureChatOpenAI to suggest an execution sequence."""
        prompt = f"Suggest an optimized sequence for executing APIs in the following OpenAPI schema:\n{openapi_data}"

        response = await self.llm_client.acompletion_create(
            messages=[{"role": "system", "content": "You are an API assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )

        return response["choices"][0]["message"]["content"].strip()

    async def extract_api_details(self, user_input):
        """Extract API method and endpoint from user input."""
        prompt = f"Extract HTTP method and API endpoint from the following request:\n{user_input}\nReturn as JSON: {{'method': '...', 'endpoint': '...'}}"
        
        response = await self.llm_client.acompletion_create(
            messages=[{"role": "system", "content": "You are an API assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        return json.loads(response["choices"][0]["message"]["content"].strip())

    async def generate_payload(self, endpoint, openapi_data):
        """Generate a sample payload based on OpenAPI schema."""
        prompt = f"Generate a JSON payload for {endpoint} based on this OpenAPI schema:\n{openapi_data}"

        response = await self.llm_client.acompletion_create(
            messages=[{"role": "system", "content": "You are an API assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )

        return json.loads(response["choices"][0]["message"]["content"].strip())
