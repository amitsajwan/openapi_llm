from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Pull the default ReAct prompt
prompt = hub.pull("hwchase17/react")

# List of tools
tools = [
    general_inquiry,
    openapi_help,
    generate_payload,
    generate_sequence,
    create_workflow,
    execute_workflow,
]

# Create the agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
