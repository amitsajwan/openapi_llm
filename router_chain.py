from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.chains.router import MultiRouteChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import json

# FastAPI App
app = FastAPI()

# Initialize LLM
llm = AzureChatOpenAI(deployment_name="your-deployment-name")

# Initialize Memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Intent Classification Prompt
intent_prompt = PromptTemplate(
    template="""
    Previous Context: {history}
    User Query: {user_input}
    Based on the conversation, classify the intent into one of:
    - general_inquiry
    - openapi_help
    - generate_payload
    - generate_sequence
    - create_workflow
    - execute_workflow
    Return the classification as JSON: {"intent": "intent_name"}
    """,
    input_variables=["history", "user_input"]
)

# Intent Chain
intent_chain = LLMChain(llm=llm, prompt=intent_prompt, memory=memory)

# Route Chains
routes = {
    "general_inquiry": LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Answer this user query: {user_input}"),
    ),
    "openapi_help": LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Provide API details based on OpenAPI spec for: {user_input}"),
    ),
    "generate_payload": LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Generate a JSON payload for: {user_input}"),
    ),
    "generate_sequence": LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Determine API execution sequence based on: {user_input}"),
    ),
    "create_workflow": LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Create a LangGraph workflow for: {user_input}"),
    ),
    "execute_workflow": LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Execute the workflow now."),
    ),
}

# MultiRouteChain
router_chain = MultiRouteChain(router_chain=intent_chain, destination_chains=routes, default_chain=routes["general_inquiry"])

# Pydantic Request Model
class UserQuery(BaseModel):
    user_input: str
    openapi_spec: dict  # Add OpenAPI spec as part of the request

@app.post("/submit_openapi")
async def handle_request(query: UserQuery):
    """Handles user queries and routes them based on intent."""
    try:
        classified_intent = await router_chain.arun({
            "user_input": query.user_input,
            "history": memory.load_memory_variables({}).get("history", "")
        })
        intent_json = json.loads(classified_intent)
        intent = intent_json.get("intent", "general_inquiry")
        
        response = await router_chain.run_chain(intent, {"user_input": query.user_input})
        memory.save_context({"user_input": query.user_input}, {"response": response})
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
