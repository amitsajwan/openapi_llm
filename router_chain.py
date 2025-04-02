from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.utils.math import cosine_similarity
from langchain.memory import ConversationBufferMemory
import json

class OpenAPIIntentRouter:
    def __init__(self, llm: AzureChatOpenAI, openapi_spec: dict, azure_api_key: str, azure_deployment_name: str):
        self.llm = llm
        self.openapi_spec = openapi_spec
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.embeddings = AzureOpenAIEmbeddings(openai_api_key=azure_api_key, deployment_name=azure_deployment_name)
        self.graph = self._initialize_router()

    def _initialize_router(self):
        """Initialize the LangGraph router with different intents."""
        intent_templates = [
            """You are an assistant that classifies user queries into intents. 
            Based on the conversation, classify the intent into one of:
            - general_inquiry
            - openapi_help
            - generate_payload
            - generate_sequence
            - create_workflow
            - execute_workflow
            Return the classification as JSON: {\"intent\": \"intent_name\"}
            Query: {query}""",
            """Extract relevant API details from OpenAPI spec: {openapi_spec}\n\nQuery: {query}""",
            """Generate a structured JSON payload using OpenAPI spec: {openapi_spec}\n\nTarget API: {query}""",
            """Determine the correct API execution order using OpenAPI spec: {openapi_spec}""",
            """Construct a LangGraph workflow based on API endpoints from OpenAPI spec: {openapi_spec}""",
            """Execute the predefined API workflow."""
        ]

        prompt_embeddings = self.embeddings.embed_documents(intent_templates)

        def prompt_router(input):
            query_embedding = self.embeddings.embed_query(input["query"])
            similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
            most_similar = intent_templates[similarity.argmax()]
            return PromptTemplate.from_template(most_similar)

        graph = StateGraph()
        graph.add_node("classify_intent", RunnableLambda(prompt_router))
        graph.add_node("generate_response", self.llm)
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "generate_response")
        return graph.compile()

    def handle_user_input(self, user_input: str):
        """Handles user queries and routes them based on intent."""
        try:
            response = self.graph.invoke({"query": user_input, "openapi_spec": self.openapi_spec})
            self.memory.save_context({"user_input": user_input}, {"response": response})
            return {"response": response}
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}
