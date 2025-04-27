# langgraph_prebuilt_agent.py
import logging
from langchain.chat_models import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from trustcall import trustcall, TrustResult
from tools import (
    set_llm_and_spec,
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

logging.basicConfig(level=logging.INFO)

class OpenApiReactRouterManager:
    def __init__(self, spec_text: str, llm=None):
        # 1) save spec & llm
        self.spec_text = spec_text
        self.llm = llm or AzureChatOpenAI(deployment_name="gpt-4", temperature=0)
        # 2) inject into tools
        set_llm_and_spec(self.llm, self.spec_text)
        # 3) build agent
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        # build structured prompt
        system = SystemMessagePromptTemplate.from_template(
            "You are an expert API assistant. Use your tools to answer user queries based on the preloaded OpenAPI spec."
        )
        human = HumanMessagePromptTemplate.from_template("User: {input}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system.template),
            ("human", human.template),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

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

        return create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=prompt,
            checkpointer=MemorySaver(),
            response_format="structured"
        )

    async def run(self, user_input: str, thread_id: str = "default-thread"):
        # wrap agent.invoke in trustcall
        args = {"input": user_input, "thread_id": thread_id}
        result = trustcall(self.agent.invoke, args, {"configurable": {"thread_id": thread_id}})
        if not result.success:
            logging.error(f"Agent call failed: {result.error}")
            raise RuntimeError(result.error)
        out = result.output
        # extract structured response or fallback
        if isinstance(out, dict) and "structured_response" in out:
            return out["structured_response"]
        if isinstance(out, dict) and "response" in out:
            return out["response"]
        return out

# Example usage (sync for demonstration; wrap in asyncio for real)
if __name__ == "__main__":
    import asyncio
    spec = open("openapi.yaml").read()
    mgr = OpenApiReactRouterManager(spec_text=spec)
    resp = asyncio.run(mgr.run("Tell me about APIs", "thread1"))
    print(resp)
    resp = asyncio.run(mgr.run("Generate API execution graph", "thread1"))
    print(resp)
    resp = asyncio.run(mgr.run("Describe execution plan", "thread1"))
    print(resp)
