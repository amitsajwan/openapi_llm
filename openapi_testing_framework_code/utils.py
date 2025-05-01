import os
import json
from typing import List, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.chat_models import AzureChatOpenAI

# Initialize global LLM
_llm = AzureChatOpenAI(
    openai_api_type="azure",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    temperature=0
)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def llm_call_helper(prompt: str, function_name: str = None) -> Any:
    response = _llm.invoke(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"response": response}

def read_swagger_files(spec_folder: str) -> List[str]:
    return [f for f in os.listdir(spec_folder) if f.endswith('.json') or f.endswith('.yaml')]
