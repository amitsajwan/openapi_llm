from openai import OpenAI

class LLMSequenceGenerator:
    def __init__(self):
        self.client = OpenAI(api_key="your-api-key")

    def generate_sequence(self, api_map):
        prompt = f"Determine execution order for APIs: {api_map}"
        response = self.client.chat.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"].split("\n")
