from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain.agents.react.agent import ReActAgent
from langchain.memory import ConversationBufferMemory

# Example tools (replace with yours)
tools = [
    Tool(name="general_inquiry", func=general_inquiry, description="Answer general questions"),
    Tool(name="openapi_help", func=openapi_help, description="Help with OpenAPI usage"),
    Tool(name="generate_payload", func=generate_payload, description="Generate JSON payloads"),
    Tool(name="generate_sequence", func=generate_sequence, description="Determine API sequence"),
    Tool(name="execute_workflow", func=execute_workflow, description="Execute an API workflow")
]

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Language model
llm = ChatOpenAI(temperature=0)

# LLM Chain with your prompt
llm_chain = LLMChain(llm=llm, prompt=react_prompt)

# Create custom ReAct agent
agent = ReActAgent(
    llm_chain=llm_chain,
    tools=tools,
    stop=["\nObservation"],
)

# Create executor with memory
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
)
