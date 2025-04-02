from langchain_openai import AzureChatOpenAI
from langchain.chains.router import MultiRouteChain
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

        intent_chain = intent_prompt | self.llm

        routes = {
            "general_inquiry": PromptTemplate.from_template("Provide a response based on the user query: {user_input}") | self.llm,
            "openapi_help": PromptTemplate.from_template("Extract relevant API details from OpenAPI spec:\n{openapi_spec}\n\nQuery: {user_input}") | self.llm,
            "generate_payload": PromptTemplate.from_template("Generate a structured JSON payload using OpenAPI spec:\n{openapi_spec}\n\nTarget API: {user_input}") | self.llm,
            "generate_sequence": PromptTemplate.from_template("Determine the correct API execution order using OpenAPI spec:\n{openapi_spec}\n\nContext: {user_input}") | self.llm,
            "create_workflow": PromptTemplate.from_template("Construct a LangGraph workflow based on API endpoints from OpenAPI spec:\n{openapi_spec}") | self.llm,
            "execute_workflow": PromptTemplate.from_template("Execute the predefined API workflow.") | self.llm,
        }

        return MultiRouteChain(
            router_chain=intent_chain,
            destination_chains=routes,
            default_chain=routes["general_inquiry"]
        )

    def handle_user_input(self, user_input: str):
        """Handles user queries and routes them based on intent."""
        try:
            classified_intent_response = self.router_chain.router_chain.invoke({
                "user_input": user_input,
                "history": self.memory.load_memory_variables({}).get("history", "")
            })

            intent_json = json.loads(classified_intent_response.strip())
            intent = intent_json.get("intent", "general_inquiry")

            response = self.router_chain.destination_chains[intent].invoke({
                "user_input": user_input,
                "openapi_spec": self.openapi_spec
            })

            self.memory.save_context({"user_input": user_input}, {"response": response})

            return {"intent": intent, "response": response}
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}
