import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from api_execution_graph_schemas import Api_execution_graph  # <- adjust if your models file name is different
from trustcall import trustcall  # <- your wrapper for safe LLM call

class OpenApiReactRouterManager:
    def __init__(self, llm, spec_text: str, llm_name: str):
        self.llm_name = llm_name
        self.llm = llm
        self.spec_text = spec_text
        self.spec = None  # you can later load YAML if needed
        self.agent = self.initialize_agent()

    def initialize_agent(self):
        tools = [
            general_query_fn,
            openapi_help_fn,
            generate_payload_fn,
            generate_api_execution_graph_fn,
            add_edge_fn,
            validate_graph_fn,
            get_execution_graph_json_fn,
            simulate_load_test_fn,
            execute_workflow_fn
        ]

        agent = create_react_agent(
            tools=tools,
            llm=self.llm,
            memory=MemorySaver(),
            prompt_template=prompt_template,
            output_type=Api_execution_graph,  # The Pydantic output model you defined
            output_format="json"  # assuming output must be JSON
        )
        return agent

    async def route(self, user_input: str, thread_id: str) -> Api_execution_graph:
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            logging.info(f"Routing user_input: {user_input} with thread_id: {thread_id}")
            response = await self.agent.ainvoke(user_input, config=config)
            logging.info(f"Received response for thread_id: {thread_id}")
            return response
        except Exception as e:
            logging.error(f"Routing failed: {e}")
            return Api_execution_graph(error="Routing failed", raw=str(e))
