# agent_full.py

import json
from pathlib import Path
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langgraph.graph import StateGraph, State
from react_router import (
    make_tool,
    general_query_fn,
    openapi_help_fn,
    generate_payload_fn,
    generate_api_execution_graph_fn
)



class OpenApiReactRouterManager:
    def __init__(self, openapi_spec_path: str, llm):
        self.spec_path = Path(openapi_spec_path)
        self.llm = llm
        self.spec_text = self.spec_path.read_text()
        self.toolkit = OpenAPIToolkit.from_spec(self.spec_path, allow_dangerous_request=True)
        self.api_tools = self.toolkit.get_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tools = self._initialize_tools()
        self.agent_executor = self._initialize_agent_executor()

    def _initialize_tools(self):
        tools = [
            make_tool(lambda x: general_query_fn(x, self.llm), "general_query", "Handle non‑API general questions."),
            make_tool(lambda x: openapi_help_fn(x, self.llm, self.spec_text), "openapi_help", "Explain OpenAPI endpoints."),
            make_tool(lambda x: generate_payload_fn(x, self.llm, self.spec_text), "generate_payload", "Create JSON payload for API."),
            make_tool(lambda x: generate_api_execution_graph_fn(x, self.llm, self.spec_text), "generate_api_execution_graph", "Produce execution graph (nodes/edges/payloads)."),
            *self.api_tools
        ]
        return tools

    def _initialize_agent_executor(self):
        system_message = SystemMessagePromptTemplate.from_template("""
You are an AI assistant for interacting with an OpenAPI. Use ONLY the available tools:
{tools}

Respond strictly in JSON with keys: response, intent, query.
Do NOT add extra commentary.
""")
        human_message = HumanMessagePromptTemplate.from_template("{input}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        json_agent = create_json_chat_agent(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            tools=self.tools,
            prompt=chat_prompt,
            stop_sequence=["Observation:"]
        )

        agent_executor = AgentExecutor(
            agent=json_agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        return agent_executor

    def invoke(self, user_input: str):
        return self.agent_executor.invoke({"input": user_input})

if __name__ == "__main__":
    llm_azure = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        openai_api_version="2023-05-15",
        temperature=0
    )
    manager = OpenApiReactRouterManager("openapi_spec.json", llm_azure)
    response = manager.invoke("Generate API execution graph for pet creation")
    print(json.dumps(response["output"], indent=2))
