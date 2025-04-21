from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Assume openapi_spec is a string containing your OpenAPI specification
openapi_spec = "..."  # Replace with your actual OpenAPI spec

@tool
def general_inquiry(user_input: str) -> str:
    prompt = PromptTemplate.from_template(
        "Answer the following general inquiry:\n\n{user_input}"
    )
    return llm.invoke(prompt.format(user_input=user_input))

@tool
def openapi_help(user_input: str) -> str:
    prompt = PromptTemplate.from_template(
        "Using the OpenAPI specification:\n\n{openapi_spec}\n\nAnswer the following question:\n\n{user_input}"
    )
    return llm.invoke(prompt.format(openapi_spec=openapi_spec, user_input=user_input))

@tool
def generate_payload(user_input: str) -> str:
    prompt = PromptTemplate.from_template(
        "Based on the OpenAPI specification:\n\n{openapi_spec}\n\nGenerate a JSON payload for the following request:\n\n{user_input}"
    )
    return llm.invoke(prompt.format(openapi_spec=openapi_spec, user_input=user_input))

@tool
def generate_sequence(user_input: str) -> str:
    prompt = PromptTemplate.from_template(
        "Using the OpenAPI specification:\n\n{openapi_spec}\n\nDetermine the sequence of API calls for the following task:\n\n{user_input}"
    )
    return llm.invoke(prompt.format(openapi_spec=openapi_spec, user_input=user_input))

@tool
def create_workflow(user_input: str) -> str:
    prompt = PromptTemplate.from_template(
        "Construct a workflow based on the OpenAPI specification:\n\n{openapi_spec}\n\nTask:\n\n{user_input}"
    )
    return llm.invoke(prompt.format(openapi_spec=openapi_spec, user_input=user_input))

@tool
def execute_workflow(user_input: str) -> str:
    prompt = PromptTemplate.from_template(
        "Execute the following workflow using the OpenAPI specification:\n\n{openapi_spec}\n\nWorkflow:\n\n{user_input}"
    )
    return llm.invoke(prompt.format(openapi_spec=openapi_spec, user_input=user_input))
