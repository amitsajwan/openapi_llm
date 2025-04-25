# File: langgraph_prebuilt_agent.py
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import ChatOpenAI
from tools import (
    general_query_fn,
    openapi_help_fn,
    generate_payload_fn,
    generate_api_execution_graph_fn,
    add_edge_fn,
    validate_graph_fn,
    describe_execution_plan_fn,
    get_execution_graph_json_fn,
)

class OpenApiReactRouterManager:
    def __init__(self, spec_text: str, llm=None):
        self.spec_text = spec_text
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        tools = [
            general_query_fn,
            openapi_help_fn,
            generate_payload_fn,
            generate_api_execution_graph_fn,
            add_edge_fn,
            validate_graph_fn,
            describe_execution_plan_fn,
            get_execution_graph_json_fn,
        ]
                # build a custom ReAct-style prompt including the OpenAPI spec
        from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        # System instruction for the agent
        system_template = """
You are an expert API testing assistant. You have access to tools to inspect OpenAPI specifications, generate payloads, build and modify execution graphs, validate graphs, and describe execution plans.
Always think step-by-step, use the ReAct format (Thought, Action, Action Input, Observation), and refer to the provided OpenAPI spec.
OpenAPI Spec:
{openapi_yaml}
"""
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        # Human input template
        human_prompt = HumanMessagePromptTemplate.from_template("User: {input}")
        custom_prompt = PromptTemplate.from_messages([system_prompt, human_prompt])

        return create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=custom_prompt,
            checkpointer=MemorySaver(),
        ),
        )

    def run(self, user_input: str, thread_id: str = "default-thread") -> str:
        inputs = {"input": user_input, "openapi_yaml": self.spec_text}
        result = self.agent.invoke(
            inputs,
            config={"configurable": {"thread_id": thread_id}}
        )
        # Extract AIMessage content
        if isinstance(result, dict) and "messages" in result:
            for m in result["messages"]:
                if isinstance(m, AIMessage):
                    return m.content
        # Fallbacks
        return result.get("response") or result.get("output") or str(result)

# Example usage
if __name__ == "__main__":
    spec = open("openapi.yaml").read()
    router = OpenApiReactRouterManager(spec_text=spec)
    print(router.run("Generate API execution graph for this spec", "thread1"))
    print(router.run("Add an edge from createPet to getPetById", "thread1"))
    print(router.run("Validate the graph", "thread1"))
    print(router.run("Describe the execution plan", "thread1"))
    print(router.run("Give me the graph as JSON", "thread1"))
