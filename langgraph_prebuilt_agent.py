# langgraph_prebuilt_agent.py
import logging
from langchain.chat_models import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import bind_trustcall, get_api_list, determine_api_sequence, generate_graph

logging.basicConfig(level=logging.INFO)

class OpenApiReactRouterManager:
    def __init__(self, spec_text: str, llm=None):
        self.spec_text = spec_text
        self.llm = llm or AzureChatOpenAI(deployment_name="gpt-4", temperature=0)
        # 1) bind global spec & LLM in tools
        from tools import _openapi_yaml, _graph_state
        _openapi_yaml = spec_text
        bind_trustcall(self.llm, [get_api_list, determine_api_sequence, generate_graph])
        # 2) build agent
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an API assistant. Use tools to answer queries on the preloaded OpenAPI spec."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        return create_react_agent(
            model=self.llm,
            tools=[get_api_list, determine_api_sequence, generate_graph],
            prompt=prompt,
            checkpointer=MemorySaver(),
            response_format="structured"
        )

    async def run(self, user_input: str, thread_id: str="thread1"):
        # TrustCall extractor handles tool selection & retries citeturn0search0
        result = extractor.invoke(user_input)  
        if not result.success:
            logging.error(f"TrustCall failed: {result.error}")
            raise RuntimeError(result.error)
        # result.output contains the structured agent response
        return result.output

# Example usage
if __name__ == "__main__":
    import asyncio
    spec = open("openapi.yaml").read()
    mgr = OpenApiReactRouterManager(spec_text=spec)
    print(asyncio.run(mgr.run("Tell me about APIs", "t1")))
    print(asyncio.run(mgr.run("Generate API execution graph", "t1")))
