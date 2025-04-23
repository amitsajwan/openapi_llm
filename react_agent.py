from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import Runnable
from langchain.tools import tool

class OpenAPIDirectRouterManager:
    def __init__(self, spec_text: str, llm, openapi_spec_path: str):
        self.spec_text = spec_text
        self.llm = llm
        self.spec_path = openapi_spec_path

        # Initialize OpenAPI tools (assume your custom OpenAPIToolkit exists)
        self.toolkit = OpenAPIToolkit.from_spec(self.spec_path, allow_dangerous_requests=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tools = self._init_llmtools()
        self.agent_executor = self._initialize_agent_executor()

    def _init_llmtools(self):
        return [
            tool("general_query", lambda x: general_query_fn(x, self.llm), return_direct=True)(description="Handle non-API general questions."),
            tool("general_help", lambda x: general_help_fn(x, self.llm, self.spec_text), return_direct=True)(description="Explain OpenAPI endpoints."),
            tool("generate_payload", lambda x: generate_payload_fn(x, self.llm, self.spec_text), return_direct=True)(description="Generate JSON payload for API."),
            tool("generate_api_execution_graph", lambda x: generate_api_execution_graph_fn(x, self.llm, self.spec_text), return_direct=True)(description="Generate execution graph from OpenAPI."),
            tool("execute_workflow", lambda x: execute_workflow_fn(x, self.llm, self.spec_text), return_direct=True)(description="Execute workflow steps based on graph."),
        ]

    def _initialize_agent_executor(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a tool selector assistant. Use the appropriate tools to assist with user queries.\n"
                       "Don’t modify the tool's output. Pass user's query to the best tool.\n"
                       "The tool will return the final response. Format: {{\"response\": ..., \"intent\": ..., \"query\": user_input}}"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

    async def handle_user_input(self, user_input: str) -> str:
        result = await self.agent_executor.ainvoke({"input": user_input})
        return result["output"]
