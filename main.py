from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_sequence_generator import LLMSequenceGenerator
from langchain_openai import AzureChatOpenAI

app = FastAPI()

# Initialize LLM
llm = AzureChatOpenAI(deployment_name="your-deployment", model="gpt-4")
sequence_generator = LLMSequenceGenerator(llm)

class UserQuery(BaseModel):
    user_input: str
    openapi_data: dict | None = None  # Optional for certain functions

@app.post("/process_query")
async def process_query(request: UserQuery):
    """Determines intent and calls the appropriate function."""
    try:
        # Step 1: Determine Intent
        intent = await sequence_generator.determine_intent(request.user_input, request.openapi_data)

        # Step 2: Call the corresponding function
        intent_mapping = {
            "suggest_sequence": sequence_generator.suggest_sequence,
            "extract_load_test_params": sequence_generator.extract_load_test_params,
            "extract_api_details": sequence_generator.extract_api_details,
            "answer_general_query": sequence_generator.answer_general_query
        }

        if intent in intent_mapping:
            response = await intent_mapping[intent](request.user_input if "extract" in intent else request.openapi_data)
            return {"intent": intent, "response": response}

        elif intent == "generate_payload":
            if not request.openapi_data:
                raise HTTPException(status_code=400, detail="OpenAPI data required for payload generation.")
            endpoint = request.user_input.split()[-1]  # Extract endpoint from input
            response = await sequence_generator.generate_payload(endpoint, request.openapi_data)
            return {"intent": intent, "response": response}

        return {"intent": "unknown", "response": "Intent not recognized."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
