from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from tools import (
    set_llm,
    general_query_fn,
    openapi_help_fn,
    generate_payload_fn,
    generate_api_execution_graph_fn,
    add_edge_fn,
    validate_graph_fn,
    describe_execution_plan_fn,
    get_execution_graph_json_fn,
    simulate_load_test_fn,
)

class OpenApiReactRouterManager:
    def __init__(self, spec_text: str, llm=None):
        self.spec_text = spec_text
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)  # LLM injection
        set_llm(self.llm)  # ensure tools use this LLM
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
            simulate_load_test_fn,
        ]

        system_template = """
You are an expert API-testing assistant. Use the tools to inspect OpenAPI specs, build/modify execution graphs, validate them, and describe plans. Always think step-by-step in ReAct format.
OpenAPI Spec:
{openapi_yaml}
"""
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_prompt = HumanMessagePromptTemplate.from_template("User: {input}")
        custom_prompt = PromptTemplate.from_messages([system_prompt, human_prompt])

        return create_react_agent(
            model=self.llm,                         # LLM citeturn0search0
            tools=tools,                            # tool set citeturn0search2
            prompt=custom_prompt,                  # custom ReAct prompt
            checkpointer=MemorySaver(),            # persist state per thread citeturn1search0
            response_format=None,
            debug=False,
        )

    def run(self, user_input: str, thread_id: str = "default-thread") -> str:
        inputs = {"input": user_input, "openapi_yaml": self.spec_text, "thread_id": thread_id}
        result = self.agent.invoke(
            inputs,
            config={"configurable": {"thread_id": thread_id}}
        )
        # extract AIMessage content
        if isinstance(result, dict) and "messages" in result:
            for m in result["messages"]:
                if isinstance(m, AIMessage):
                    return m.content
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
    print(router.run("Simulate load test with 10 users", "thread1"))
