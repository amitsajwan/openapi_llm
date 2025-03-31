import json
from fastapi import FastAPI, HTTPException
from langchain_openai import AzureChatOpenAI
from langchain_core.memory import ConversationBufferMemory
from llm_sequence_generator import LLMSequenceGenerator

app = FastAPI()

# Initialize LLM and Memory
llm = AzureChatOpenAI()
memory = ConversationBufferMemory()
generator = LLMSequenceGenerator(llm, memory)

@app.post("/chat")
async def chat(user_input: str):
    """Handles user queries by determining intent and executing the relevant function."""
    intent = await generator.determine_intent(user_input)
    
    if "suggest_sequence" in intent:
        return {"sequence": await generator.suggest_sequence(user_input)}
    elif "extract_load_test_params" in intent:
        return {"params": await generator.extract_load_test_params(user_input)}
    elif "extract_api_details" in intent:
        return {"details": await generator.extract_api_details(user_input)}
    elif "generate_payload" in intent:
        return {"payload": await generator.generate_payload(user_input)}
    else:
        return {"response": await generator.answer_general_query(user_input)}

@app.get("/memory")
async def get_memory():
    """Returns conversation memory for debugging and state tracking."""
    return memory.load_memory_variables({})
