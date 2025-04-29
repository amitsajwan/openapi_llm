import logging
from typing import Optional

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from trust_call import call_llm

from api_execution_graph_schema import (
    Api_execution_graph,
    GraphOutput,
    TextContent,
)

logging.basicConfig(level=logging.INFO)


class ApiGraphManager:
    llm: AzureChatOpenAI = None
    openapi_yaml: str = None
    graph_state: Api_execution_graph = Api_execution_graph()

    @classmethod
    def set_llm(cls, llm: AzureChatOpenAI, openapi_yaml: str):
        """Initialize the LLM instance and spec for all tools."""
        cls.llm = llm
        cls.openapi_yaml = openapi_yaml
        logging.info(f"LLM set to {cls.llm}")

    ############################################################################
    # 1) Generate Execution Graph
    ############################################################################
    @staticmethod
    @tool
    def generate_api_execution_graph_fn(user: str) -> Api_execution_graph:
        """Generates an API execution graph (nodes & edges) from OpenAPI spec."""
        logging.info("Starting generate_api_execution_graph_fn")
        prompt_template = """
1. Parse the provided OpenAPI spec (YAML/JSON):
{openapi_spec}

User's query: {user_input}
2. Identify all operations (paths, methods) and their dependencies.
3. Generate example payloads for POST and PUT, resolving $ref, enums, and nested schemas.
4. Suggest parallel execution only when operations are independent.
5. Include 'verify' nodes after each state-changing call to check resource integrity.
6. OperationId should be like createProduct, updateProduct, getAllAfterCreate, verifyAfterUpdate, etc.
7. Edge will not go back to earlier node; if we need to do an operation again it will be new node with same operation, but id will be different.
8. Flow should start with dummy Start Node (operationId=START) and end at End node (operationId=END).
9. Add proper text content explaining the solution or execution plan in user friendly way.
10. Verification: Do we have proper edges.
11. Do we have START and END.
12. Text content is proper.
13. Payloads added for relevant operations.
"""
        formatted = prompt_template.format(
            openapi_spec=ApiGraphManager.openapi_yaml, user_input=user
        )
        ApiGraphManager.graph_state = call_llm(
            formatted, ApiGraphManager.llm, Api_execution_graph
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 2) Add Edge
    ############################################################################
    @staticmethod
    @tool
    def add_edge_fn(from_node: str, to_node: str) -> Api_execution_graph:
        """Add an edge to the existing API execution graph."""
        logging.info(f"Adding edge from {from_node} to {to_node}")
        prompt = f"""
Given the existing API execution graph {ApiGraphManager.graph_state.graph.model_dump()},
Add an edge from {from_node} to {to_node}.
Ensure:
- Flow still starts at START and ends at END.
- No cycles (DAG).
- Text content remains clear.
"""
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, Api_execution_graph
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 3) Validate Graph
    ############################################################################
    @staticmethod
    @tool
    def validate_graph_fn() -> Api_execution_graph:
        """Validate the current API execution graph."""
        logging.info("Starting validate_graph_fn")
        prompt = f"Validate the current API execution graph {ApiGraphManager.graph_state.graph}."
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, Api_execution_graph
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 4) Get JSON
    ############################################################################
    @staticmethod
    @tool
    def get_execution_graph_json_fn() -> Api_execution_graph:
        """Get the JSON representation of the current API execution graph."""
        logging.info("Starting get_execution_graph_json_fn")
        prompt = "Get the JSON representation of the current API execution graph."
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, Api_execution_graph
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 5) General Query
    ############################################################################
    @staticmethod
    @tool
    def general_query_fn(user: str) -> Api_execution_graph:
        """Answer any general question using LLM reasoning."""
        logging.info("Starting general_query_fn")
        prompt = f"Based on your understanding, answer user query: {user}"
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, TextContent
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 6) OpenAPI Help
    ############################################################################
    @staticmethod
    @tool
    def openapi_help_fn(user: str) -> Api_execution_graph:
        """Explain OpenAPI endpoints. Answers user query for given OpenAPI swagger."""
        logging.info("Starting openapi_help_fn")
        prompt = f"Based on the OpenAPI specification, answer user query: {user}"
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, TextContent
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 7) Generate Payload
    ############################################################################
    @staticmethod
    @tool
    def generate_payload_fn(user: str) -> Api_execution_graph:
        """Generate a realistic JSON payload matching the given OpenAPI schema."""
        logging.info("Starting generate_payload_fn")
        prompt = f"Given the OpenAPI specification, generate a JSON payload for: {user}"
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, TextContent
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 8) Simulate Load Test
    ############################################################################
    @staticmethod
    @tool
    def simulate_load_test_fn(num_users: int = 1) -> Api_execution_graph:
        """Simulate a load test by generating concurrent execution plan for N users."""
        logging.info("Starting simulate_load_test_fn")
        prompt = f"Based on the current API execution graph {ApiGraphManager.graph_state.graph.model_dump()}, simulate a load test with {num_users} users."
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, TextContent
        )
        return ApiGraphManager.graph_state

    ############################################################################
    # 9) Execute Workflow
    ############################################################################
    @staticmethod
    @tool
    def execute_workflow_fn(user: str) -> Api_execution_graph:
        """Execute workflow. Summarize the execution plan of OpenAPI."""
        logging.info("Starting execute_workflow_fn")
        prompt = f"Summarize execution plan: {ApiGraphManager.graph_state.graph.model_dump()}"
        ApiGraphManager.graph_state = call_llm(
            prompt, ApiGraphManager.llm, TextContent
        )
        return ApiGraphManager.graph_state
        
