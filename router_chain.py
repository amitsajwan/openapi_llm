from langchain.chains import RouterChain, LLMChain
from langchain_core.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiRouteChain
import json

class OpenAPIIntentRouter:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.router_chain = self._initialize_router()

    def _initialize_router(self):
        """Initialize the RouterChain with different intents."""
        intent_prompt = PromptTemplate(
            template="""
            Previous Context: {history}
            User Query: {user_input}
            Based on the conversation, classify the intent into one of:
            - general_inquiry
            - openapi_help
            - generate_payload
            - generate_sequence
            - create_workflow
            - execute_workflow
            Return the classification as JSON: {"intent": "intent_name"}
            """,
            input_variables=["history", "user_input"]
        )
        
        intent_chain = LLMChain(llm=self.llm, prompt=intent_prompt, memory=self.memory)

        routes = {
            "general_inquiry": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Answer this user query: {user_input}"),
            ),
            "openapi_help": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Provide API details based on OpenAPI spec for: {user_input}"),
            ),
            "generate_payload": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Generate a JSON payload for: {user_input}"),
            ),
            "generate_sequence": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Determine API execution sequence based on: {user_input}"),
            ),
            "create_workflow": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Create a LangGraph workflow for: {user_input}"),
            ),
            "execute_workflow": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Execute the workflow now."),
            ),
        }

        return RouterChain(intent_chain, routes)

    async def handle_user_input(self, user_input: str):
        """Processes user input and routes it to the appropriate chain."""
        classified_intent = await self.router_chain.run({"user_input": user_input, "history": self.memory.load_memory_variables({}).get("history", "")})
        
        intent_json = json.loads(classified_intent)
        intent = intent_json.get("intent", "general_inquiry")
        
        response = await self.router_chain.route(intent, {"user_input": user_input})
        self.memory.save_context({"user_input": user_input}, {"response": response})
        
        return response
