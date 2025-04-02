from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain.chains.router import MultiRouteChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.utils.math import cosine_similarity
import json

class OpenAPIIntentRouter:
    def __init__(self, llm: AzureChatOpenAI, openapi_spec: dict):
        self.llm = llm
        self.openapi_spec = openapi_spec
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.embeddings = OpenAIEmbeddings()
        self.router_chain = self._initialize_router()

    def _initialize_router(self):
        """Initialize the MultiRouteChain with different intents."""
        intent_templates = [
            """Classify intent based on conversation:
            - general_inquiry
            - openapi_help
            - generate_payload
            - generate_sequence
            - create_workflow
            - execute_workflow
            Return JSON: {"intent": "intent_name"}.
            
            Query: {query}""",
            """Extract API details from OpenAPI spec: {openapi_spec}\n\nQuery: {query}""",
            """Generate a structured JSON payload using OpenAPI spec: {openapi_spec}\n\nTarget API: {query}""",
            """Determine correct API execution order using OpenAPI spec: {openapi_spec}""",
            """Construct a LangGraph workflow from OpenAPI spec: {openapi_spec}""",
            """Execute the predefined API workflow."""
        ]

        intent_embeddings = self.embeddings.embed_documents(intent_templates)

        def route_intent(input):
            query_embedding = self.embeddings.embed_query(input["query"])
            similarity = cosine_similarity([query_embedding], intent_embeddings)[0]
            best_match = intent_templates[similarity.argmax()]
            return PromptTemplate.from_template(best_match)

        return MultiRouteChain(
            router_chain=RunnableLambda(route_intent),
            destination_chains={
                "general_inquiry": RunnableLambda(lambda x: self.llm.invoke(x["query"])),
                "openapi_help": RunnableLambda(lambda x: self.llm.invoke(x["query"])),
                "generate_payload": RunnableLambda(lambda x: self.llm.invoke(x["query"])),
                "generate_sequence": RunnableLambda(lambda x: self.llm.invoke(x["query"])),
                "create_workflow": RunnableLambda(lambda x: self.llm.invoke(x["query"])),
                "execute_workflow": RunnableLambda(lambda x: self.llm.invoke(x["query"]))
            },
            default_chain=RunnableLambda(lambda x: self.llm.invoke(x["query"]))
        )

    def handle_user_input(self, user_input: str):
        """Handles user queries and routes them based on intent."""
        try:
            intent_response = self.router_chain.router_chain.invoke({"query": user_input})
            response = self.router_chain.invoke({"query": user_input, "openapi_spec": self.openapi_spec}, config={"destination": intent_response})
            self.memory.save_context({"user_input": user_input}, {"response": response})
            return {"intent": intent_response, "response": response}
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}
            
