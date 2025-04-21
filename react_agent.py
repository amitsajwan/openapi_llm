import json
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.memory import ConversationBufferMemory

# 1. Prepare your existing tools (example placeholders)
from your_tools import (
    general_inquiry,
    openapi_help,
    generate_payload,
    generate_sequence,
    create_workflow,
    execute_workflow,
)
tools = [
    general_inquiry,
    openapi_help,
    generate_payload,
    generate_sequence,
    create_workflow,
    execute_workflow,
]

# 2. Build a strict JSON system prompt
system_template = """
You are an OpenAPI assistant. Use ONLY the available tools to answer user queries.
Respond strictly in JSON with keys: {"response": "...", "intent": "...", "query": "..."}.
Do NOT add any extra commentary or summaries.
"""
system_msg = SystemMessagePromptTemplate.from_template(system_template)
human_msg  = HumanMessagePromptTemplate.from_template("{input}")
prompt     = ChatPromptTemplate.from_messages([system_msg, human_msg])

# 3. Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 4. Create the JSON Chat Agent
agent = create_json_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=["Observation:"],  # prevent extra text
)  # 

# 5. Attach Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # :contentReference[oaicite:3]{index=3}

# 6. Build Executor with Memory
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 7. Invoke the agent in a multi‑turn chat
# First turn
result1 = agent_executor.invoke({"input": "List all pets."})
print(result1["output"])  # -> dict with response, intent, query

# Second turn (memory retained)
result2 = agent_executor.invoke({"input": "Now generate API execution graph for listing pets."})
print(result2["output"])
