import json
from langchain_openai import AzureChatOpenAI
from langchain_core.memory import ConversationBufferMemory

class LLMSequenceGenerator:
    def __init__(self, llm: AzureChatOpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory
    async def determine_intent(self, user_input: str):
    """Determines the intent of the user query with memory awareness."""
    past_messages = self.memory.load_memory_variables({}).get("history", "")

    prompt = f"""
    Previous Context: {past_messages}
    User Query: {user_input}
    Based on previous conversation, determine if the user is continuing a past intent 
    or switching to a new topic. Return only the intent.
    """
    
    return await self.llm.invoke(prompt)

    async def suggest_sequence(self, user_input: str):
        """Suggests an execution sequence for APIs based on dependencies."""
        prompt = f"Suggest an execution sequence based on: {user_input}"
        response = await self.llm.ainvoke(prompt)
        sequence = json.loads(response)
        self.memory.save_context({"user_input": user_input}, {"sequence": sequence})
        return sequence
    
    async def extract_load_test_params(self, user_input: str):
        """Extracts load test parameters from user input."""
        prompt = f"Extract load test parameters (num_users, duration) from: {user_input}"
        response = await self.llm.ainvoke(prompt)
        try:
            params = json.loads(response)
            self.memory.save_context({"user_input": user_input}, {"params": params})
            return params
        except json.JSONDecodeError:
            return None
    
    async def extract_api_details(self, user_input: str):
        """Extracts API method and endpoint from user input."""
        prompt = f"Extract API details (method, endpoint) from: {user_input}"
        response = await self.llm.ainvoke(prompt)
        try:
            details = json.loads(response)
            self.memory.save_context({"user_input": user_input}, {"details": details})
            return details
        except json.JSONDecodeError:
            return None
    
    async def generate_payload(self, user_input: str):
        """Generates a payload for a given API endpoint."""
        prompt = f"Generate a JSON payload based on: {user_input}"
        response = await self.llm.ainvoke(prompt)
        payload = json.loads(response)
        self.memory.save_context({"user_input": user_input}, {"payload": payload})
        return payload
    
    async def answer_general_query(self, user_input: str):
        """Answers general queries that do not match specific intents."""
        prompt = f"Answer the following general query: {user_input}"
        response = await self.llm.ainvoke(prompt)
        self.memory.save_context({"user_input": user_input}, {"response": response.strip()})
        return response.strip()
