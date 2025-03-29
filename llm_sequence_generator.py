import asyncio
from api_executor import APIExecutor
from llm_sequence_generator import LLMSequenceGenerator

class APIWorkflowManager:
    def __init__(self, openapi_data, llm_client):
        self.openapi_data = openapi_data
        self.executor = APIExecutor(openapi_data)
        self.llm = LLMSequenceGenerator(llm_client)
    
    async def execute_workflow(self, user_input):
        """Determines execution sequence and executes API calls."""
        intent = self.llm.determine_intent(user_input, self.openapi_data)
        
        if intent == "run_sequence":
            sequence = self.llm.suggest_sequence(self.openapi_data)
            user_confirmation = await self.get_user_confirmation(sequence)
            if not user_confirmation:
                return {"error": "Execution cancelled by user."}
            return await self.run_sequence(sequence)
        
        elif intent == "load_test":
            num_users, duration = self.llm.extract_load_test_params(user_input)
            return await self.run_load_test(num_users, duration)
        
        return {"error": "Invalid request."}
    
    async def get_user_confirmation(self, sequence):
        """Simulates user confirmation for execution sequence."""
        print(f"Suggested execution sequence: {sequence}")
        return True  # Assume user confirms for now
    
    async def run_sequence(self, sequence):
        """Executes API calls in a defined sequence."""
        results = {}
        for step in sequence:
            method, endpoint = step["method"], step["endpoint"]
            payload = step.get("payload", None)
            results[endpoint] = await self.executor.execute_api(method, endpoint, payload)
        return results
    
    async def run_load_test(self, num_users, duration):
        """Executes API load testing."""
        return await self.executor.execute_load_test(num_users, duration)
