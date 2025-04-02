from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain.chains.router import MultiRouteChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.utils.math import cosine_similarity
import json

class OpenAPIIntentRouter:
    def __init__(self, llm: AzureChatOpenAI, openapi_spec: dict):
        self.llm = llm
        self.openapi_spec = openapi_spec
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.embeddings = OpenAIEmbeddings()
        self.prompt_templates = self._initialize_prompts()
        self.prompt_embeddings = self.embeddings.embed_documents(self.prompt_templates)
        self.router_chain = self._initialize_router()

    def _initialize_prompts(self):
        """Defines intent-based prompts."""
        return [
            "General Inquiry: Answer general API-related questions.",
            "OpenAPI Help: Extract relevant API details from OpenAPI spec.",
            "Generate Payload: Generate a structured JSON payload using OpenAPI spec.",
            "Generate Sequence: Determine the correct API execution order using OpenAPI spec.",
            "Create Workflow: Construct a LangGraph workflow based on API endpoints.",
            "Execute Workflow: Execute the predefined API workflow."
        ]

    def _select_prompt(self, user_query):
        """Selects the most relevant prompt using cosine similarity."""
        query_embedding = self.embeddings.embed_query(user_query)
        similarity = cosine_similarity([query_embedding], self.prompt_embeddings)[0]
        return self.prompt_templates[similarity.argmax()]

    def _initialize_router(self):
        """Initialize the MultiRouteChain with different intents."""
        def prompt_router(input):
            selected_prompt = self._select_prompt(input["user_input"])
            return PromptTemplate.from_template(selected_prompt)

        routes = {
            "general_inquiry": PromptTemplate.from_template("General API question: {user_input}") | self.llm,
            "openapi_help": PromptTemplate.from_template("Extract API details: {openapi_spec}\n\nQuery: {user_input}") | self.llm,
            "generate_payload": PromptTemplate.from_template("Generate JSON payload: {openapi_spec}\n\nTarget API: {user_input}") | self.llm,
            "generate_sequence": PromptTemplate.from_template("Determine execution order: {openapi_spec}") | self.llm,
            "create_workflow": PromptTemplate.from_template("Construct workflow using OpenAPI spec: {openapi_spec}") | self.llm,
            "execute_workflow": PromptTemplate.from_template("Execute API workflow.") | self.llm,
        }

        return MultiRouteChain(
            router_chain=RunnableLambda(prompt_router),
            destination_chains=routes,
            default_chain=routes["general_inquiry"],
        )

    def handle_user_input(self, user_input: str):
        """Handles user queries and routes them based on intent."""
        try:
            response = self.router_chain.invoke({
                "user_input": user_input,
                "openapi_spec": self.openapi_spec
            })
            self.memory.save_context({"user_input": user_input}, {"response": response})
            return response
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}
