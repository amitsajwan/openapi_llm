# react_router.py

from langchain.agents import Tool

def make_tool(func, name, desc):
    return Tool.from_function(func=func, name=name, description=desc)

def general_query_fn(user_input: str, llm) -> dict:
    response = llm.invoke(f"Answer this general question:\n\n{user_input}")
    return {"response": response, "intent": "general_query", "query": user_input}

def openapi_help_fn(user_input: str, llm, spec_text: str) -> dict:
    response = llm.invoke(f"Using this OpenAPI spec:\n{spec_text}\n\nQuestion:\n{user_input}")
    return {"response": response, "intent": "openapi_help", "query": user_input}

def generate_payload_fn(user_input: str, llm, spec_text: str) -> dict:
    response = llm.invoke(f"Generate a JSON payload for:\n{user_input}\n\nSpec:\n{spec_text}")
    return {"response": response, "intent": "generate_payload", "query": user_input}

def generate_api_execution_graph_fn(user_input: str, llm, spec_text: str) -> dict:
    response = llm.invoke(
        "Determine the API execution graph:\n"
        f"{user_input}\n\nSpec:\n{spec_text}\n\n"
        "Return JSON with TextContent and Graph {nodes, edges}."
    )
    return {"response": response, "intent": "generate_api_execution_graph", "query": user_input}
