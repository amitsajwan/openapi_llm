from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import ChatOpenAI
from tools import (
    general_query_fn,
    openapi_help_fn,
    generate_payload_fn,
    generate_api_execution_graph_fn,
    add_edge_fn,
    describe_execution_plan_fn,
    get_execution_graph_json_fn,
    validate_graph_fn,
    execute_workflow_fn,  # Add this tool
)

class OpenApiReactRouterManager:
    def __init__(self, spec_text: str, llm):
        self.spec_text = spec_text
        self.llm = llm
        self.execution_graph = None
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        tools = [
            lambda query: general_query_fn(query, self.llm),
            lambda question: openapi_help_fn(question, self.spec_text, self.llm),
            lambda endpoint, schema: generate_payload_fn(endpoint, schema, self.llm),
            lambda user_input: generate_api_execution_graph_fn(user_input, self.spec_text, self.llm),
            lambda user_instruction: add_edge_fn(user_instruction, self.llm),
            lambda graph: describe_execution_plan_fn(graph, self.llm),
            lambda _: get_execution_graph_json_fn(),
            lambda graph: validate_graph_fn(graph, self.llm),
            lambda _: execute_workflow_fn("", self.llm),  # Add execute workflow tool
        ]
        return create_react_agent(
            llm=self.llm,
            tools=tools,
            checkpointer=MemorySaver()
        )

    def run(self, user_input: str, thread_id: str = "default-thread") -> str:
        try:
            result = self.agent.invoke(
                {"input": user_input},
                config={"configurable": {"thread_id": thread_id}}
            )
            return result.get("response", "No response")
        except Exception as e:
            return f"Error running agent: {str(e)}"

    def handle_user_message(self, user_input: str, thread_id: str) -> str:
        try:
            out = self.agent.invoke(
                {"input": user_input},
                config={"configurable": {"thread_id": thread_id}}
            )
            return out.get("response", "No response")
        except Exception as e:
            return f"Error handling user message: {str(e)}"

if __name__ == "__main__":
    tid = "user-session-1"
    llm = ChatOpenAI()
    manager = OpenApiReactRouterManager(spec_text="Your API spec here", llm=llm)

    # Test the execute workflow intent
    print(manager.handle_user_message("Execute workflow based on current graph", tid))
