from langchain_core.tools import tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder

# 1) Define function-tools
@tool()
def generate_api_execution_graph(spec_text: str) -> dict:
    # … your existing graph-gen logic …
    return {"nodes": nodes, "edges": edges}

# 2) Build prompt
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message="You are an API orchestration assistant.",
    extra_prompt_messages=[MessagesPlaceholder("chat_history", optional=True)],
)

# 3) Instantiate agent
llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
agent = create_openai_functions_agent(llm=llm, tools=[generate_api_execution_graph], prompt=prompt)

# 4) Execute
agent_executor = AgentExecutor(agent=agent, tools=[generate_api_execution_graph], verbose=True)
resp = agent_executor.invoke({"input": "Build execution graph from this spec", "spec_text": petstore_yaml})

# 5) Raw JSON result
graph = resp["output"]["content"]
print("Graph JSON:", graph)
