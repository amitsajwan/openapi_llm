from typing import Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# 1. Load or define your OpenAPI spec
openapi_spec: Dict[str, Any] = {
    # Replace this with your actual OpenAPI specification
}

# 2. Define a custom ReAct prompt
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use this format for your reasoning:

Question: the input question you must answer
Thought: you should think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# 3. Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 4. Define tool functions
def general_inquiry_tool(user_input: str) -> str:
    prompt = f"Answer the following general inquiry:\n\n{user_input}"
    return llm.invoke(prompt)

def openapi_help_tool(user_input: str) -> str:
    prompt = f"Using the OpenAPI specification:\n\n{openapi_spec}\n\nAnswer the following question:\n\n{user_input}"
    return llm.invoke(prompt)

def generate_payload_tool(user_input: str) -> str:
    prompt = f"Based on the OpenAPI specification:\n\n{openapi_spec}\n\nGenerate a JSON payload for the following request:\n\n{user_input}"
    return llm.invoke(prompt)

def generate_sequence_tool(user_input: str) -> str:
    prompt = f"Using the OpenAPI specification:\n\n{openapi_spec}\n\nDetermine the sequence of API calls for the following task:\n\n{user_input}"
    return llm.invoke(prompt)

def create_workflow_tool(user_input: str) -> str:
    prompt = f"Construct a workflow based on the OpenAPI specification:\n\n{openapi_spec}\n\nTask:\n\n{user_input}"
    return llm.invoke(prompt)

def execute_workflow_tool(user_input: str) -> str:
    prompt = f"Execute the following workflow using the OpenAPI specification:\n\n{openapi_spec}\n\nWorkflow:\n\n{user_input}"
    return llm.invoke(prompt)

# 5. Wrap tools with metadata
tools = [
    Tool(name="general_inquiry", func=general_inquiry_tool, description="Handle general inquiries."),
    Tool(name="openapi_help", func=openapi_help_tool, description="Provide help regarding OpenAPI specifications."),
    Tool(name="generate_payload", func=generate_payload_tool, description="Generate a payload based on input."),
    Tool(name="generate_sequence", func=generate_sequence_tool, description="Generate a sequence based on input."),
    Tool(name="create_workflow", func=create_workflow_tool, description="Create a workflow based on input."),
    Tool(name="execute_workflow", func=execute_workflow_tool, description="Execute a workflow based on input."),
]

# 6. Create the ReAct agent
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# 7. Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 8. Handle user input
def handle_user_input(user_input: str) -> Dict[str, Any]:
    response = agent_executor.invoke({"input": user_input})
    return {"response": response}

# Example usage:
# result = handle_user_input("Generate API execution graph for petstore")
# print(result["response"])
# result = await router.handle_user_input("List all endpoints", [])
# print(result)
