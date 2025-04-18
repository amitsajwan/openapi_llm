from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langgraph.graph import State  # ✅ Define State
import json

# ✅ Define State Schema
class RouterState(State):
    query: str
    intent: str = ""
    history: list = []

class OpenAPIIntentRouter:
    def __init__(self, llm: AzureChatOpenAI, openapi_spec: dict):
        self.llm = llm
        self.openapi_spec = openapi_spec
        self.graph = self._initialize_router()

    def _initialize_router(self):
        """Initialize the LangGraph router with different intents."""
        
        def classify_intent(state: RouterState) -> RouterState:
            """Classifies user query into an intent using AzureChatOpenAI."""
            query = state.query
            history = state.history

            classification_prompt = PromptTemplate.from_template("""
                You are an AI assistant that classifies user queries into intents.
                Based on the conversation history, classify the intent into one of:
                - general_inquiry
                - openapi_help
                - generate_payload
                - generate_sequence
                - create_workflow
                - execute_workflow
                
                Return the classification as JSON: {{"intent": "intent_name"}}
                
                Conversation history: {history}
                Query: {query}
            """).format(query=query, history=history)

            response = self.llm.invoke(classification_prompt)
            intent = json.loads(response)["intent"]

            return RouterState(query=query, intent=intent, history=history)

        def generate_response(state: RouterState) -> RouterState:
            """Generates a response based on the classified intent."""
            intent = state.intent
            query = state.query
            history = state.history

            prompt_templates = {
                "general_inquiry": "Answer general questions: {query}",
                "openapi_help": "Extract API details from OpenAPI spec: {openapi_spec}\n\nQuery: {query}",
                "generate_payload": "Generate a structured JSON payload using OpenAPI spec: {openapi_spec}\n\nTarget API: {query}",
                "generate_sequence": "Determine the correct API execution order using OpenAPI spec: {openapi_spec}",
                "create_workflow": "Construct a LangGraph workflow based on API endpoints from OpenAPI spec: {openapi_spec}",
                "execute_workflow": "Execute the predefined API workflow."
            }

            if intent not in prompt_templates:
                return RouterState(query=query, intent=intent, history=history)

            selected_prompt = PromptTemplate.from_template(prompt_templates[intent])
            formatted_prompt = selected_prompt.format(openapi_spec=self.openapi_spec, query=query)

            response = self.llm.invoke(formatted_prompt)

            # Append conversation to history
            updated_history = history + [{"user": query, "bot": response}]

            return RouterState(query=query, intent=intent, history=updated_history)

        # ✅ Define LangGraph with RouterState
        graph = StateGraph(RouterState)
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("generate_response", generate_response)

        # ✅ Define Execution Flow
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "generate_response")

        return graph.compile()

    def handle_user_input(self, user_input: str, history=None):
        """Handles user queries and routes them based on intent, keeping conversation history."""
        if history is None:
            history = []

        try:
            state = RouterState(query=user_input, history=history)
            response = self.graph.invoke(state)
            return response.dict()  # ✅ Convert State object to dict for output
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}


def handle_user_input(self, user_input: str, history: list = []):
    """Handles user queries and routes them based on intent."""
    try:
        # Initialize state with user input & history
        state = RouterState(query=user_input, history=history)

        # Invoke the state graph (RouterState object is returned)
        updated_state = self.graph.invoke(state)

        # ✅ Extract the latest bot response correctly
        if updated_state.history and isinstance(updated_state.history[-1], dict):
            latest_response = updated_state.history[-1].get("bot", "No response generated.")
        else:
            latest_response = "No response generated."

        return {"response": latest_response}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}
        
def handle_user_input(self, user_input: str, history: list = []):
    """Handles user queries and routes them based on intent."""
    try:
        # Initialize state with user input & history
        state = RouterState(query=user_input, history=history)

        # 🚀 If graph.invoke is async, ensure it's awaited
        if callable(getattr(self.graph, "ainvoke", None)):
            updated_state = await self.graph.ainvoke(state)
        else:
            updated_state = self.graph.invoke(state)

        # ✅ Ensure `history` is a list
        if not isinstance(updated_state.history, list):
            return {"error": "Invalid response format from state graph."}

        # ✅ Extract the latest bot response safely
        latest_entry = updated_state.history[-1] if updated_state.history else {}
        latest_response = latest_entry.get("bot", "No response generated.")

        return {"response": latest_response}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}
  def handle_user_input(self, user_input: str, history: list = []):
    """Handles user queries and routes them based on intent."""
    try:
        # 🚀 Ensure correct input format
        state = {"query": user_input, "history": history}

        # 🔄 Handle async or sync execution
        if callable(getattr(self.graph, "ainvoke", None)):
            updated_state = await self.graph.ainvoke(state)
        else:
            updated_state = self.graph.invoke(state)

        # ✅ Access history safely
        history = updated_state["history"] if "history" in updated_state else []
        
        # ✅ Get latest bot response safely
        latest_response = history[-1]["bot"] if history else "No response generated."

        return {"response": latest_response}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}
        
