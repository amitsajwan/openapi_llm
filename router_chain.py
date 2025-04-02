from langchain_openai import AzureChatOpenAI
from langchain.chains.router import MultiRouteChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import json

class OpenAPIIntentRouter:
    def __init__(self, llm: AzureChatOpenAI, openapi_spec: dict):
        self.llm = llm
        self.openapi_spec = openapi_spec
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.router_chain = self._initialize_router()

    def _initialize_router(self):
        """Initialize the MultiRouteChain with different intents."""
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
                prompt=PromptTemplate.from_template("Answer the following user query: {user_input}"),
            ),
            "openapi_help": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Provide relevant API details from the OpenAPI spec:\n{openapi_spec}\n\nUser Query: {user_input}"),
            ),
            "generate_payload": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Generate a valid JSON payload using the OpenAPI spec:\n{openapi_spec}\n\nFor API: {user_input}"),
            ),
            "generate_sequence": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Analyze dependencies and determine the correct API execution sequence using the OpenAPI spec:\n{openapi_spec}\n\nFor APIs related to: {user_input}"),
            ),
            "create_workflow": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Design a LangGraph workflow utilizing the API endpoints from the OpenAPI spec:\n{openapi_spec}"),
            ),
            "execute_workflow": LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template("Initiate and execute the predefined workflow."),
            ),
        }

        return MultiRouteChain(
            router_chain=intent_chain,
            destination_chains=routes,
            default_chain=routes["general_inquiry"]
        )

    def handle_user_input(self, user_input: str):
        """Handles user queries and routes them based on intent."""
        try:
            classified_intent_response = self.router_chain.router_chain.run({
                "user_input": user_input,
                "history": self.memory.load_memory_variables({}).get("history", "")
            })

            intent_json = json.loads(classified_intent_response.strip())
            intent = intent_json.get("intent", "general_inquiry")

            response = self.router_chain.destination_chains[intent].run({
                "user_input": user_input,
                "openapi_spec": self.openapi_spec
            })

            self.memory.save_context({"user_input": user_input}, {"response": response})

            return {"intent": intent, "response": response}
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}
