from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
import asyncio

# Import the intent router
from openapi_intent_router import OpenAPIIntentRouter

app = FastAPI()

# Initialize LLM and Intent Router
llm = AzureChatOpenAI(deployment_name="your-deployment-name")
intent_router = OpenAPIIntentRouter(llm)

class UserQuery(BaseModel):
    user_input: str
    openapi_spec: dict  # Add OpenAPI spec as part of the request

@app.post("/submit_openapi")
async def handle_request(query: UserQuery):
    """Handles user queries and routes them based on intent."""
    try:
        response = await intent_router.handle_user_input(query.user_input, query.openapi_spec)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
