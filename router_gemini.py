import os
import json
import logging
from typing import Any, Dict, List, Optional

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
import networkx as nx
from pydantic import BaseModel, Field

# Assuming trust_call.py is in the same directory
try:
    from trust_call import call_llm
except ImportError:
    print("Error: trust_call.py not found. Please ensure it's in the same directory.")
    exit(1)

# Assuming API schema definitions are in these files
try:
    from api_execution_graph_schema import (
        Api_execution_graph,
        GraphOutput,
        TextContent,
        Graph,
        Node
    )
    from graph_utils.operations import GraphOps
except ImportError:
    print("Error: api_execution_graph_schema.py or graph_utils/operations.py not found.")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Define path for caching
payload_path = "./cache/payloads.json"
schema_path = "./cache/schema.json"

# --- State Schema ---
class RouterState(BaseModel):
    user_input: Optional[str] = None
    next_step: Optional[str] = None
    plan: Optional[List[str]] = None
    openapi_schema: Optional[dict] = None
    execution_graph: Optional[GraphOutput] = None
    payloads: Optional[dict] = None
    response: Optional[str] = None
    action_history: List[str] = Field(default_factory=list)  # To track past actions

# --- Graph Utilities (Moved Here) ---
class GraphOps:
    @staticmethod
    def add_edge(graph: dict, src: str, dest: str) -> dict:
        if src not in graph:
            raise ValueError(f"Source node '{src}' does not exist.")
        if dest not in graph:
            raise ValueError(f"Destination node '{dest}' does not exist.")

        if "next" not in graph[src]:
            graph[src]["next"] = []
        if dest not in graph[src]["next"]:
            graph[src]["next"].append(dest)
        return graph

    @staticmethod
    def remove_node(graph: dict, node: str) -> dict:
        if node not in graph:
            raise ValueError(f"Node '{node}' not found.")
        del graph[node]
        for _, data in graph.items():
            if "next" in data and node in data["next"]:
                data["next"].remove(node)
        return graph

    @staticmethod
    def remove_edge(graph: dict, src: str, dest: str) -> dict:
        if src not in graph or "next" not in graph[src]:
            raise ValueError(f"Source node '{src}' not found or has no outgoing edges.")
        if dest in graph[src]["next"]:
            graph[src]["next"].remove(dest)
        return graph

    @staticmethod
    def rename_node_in_edges(graph: dict, old_name: str, new_name: str) -> dict:
        for node, data in graph.items():
            if "next" in data:
                graph[node]["next"] = [new_name if n == old_name else n for n in data["next"]]
        return graph

# --- Graph Validator (Moved Here) ---
def validate_graph(graph: dict) -> tuple[bool, str]:
    try:
        G = nx.DiGraph()
        for node, data in graph.items():
            for nxt in data.get("next", []):
                G.add_edge(node, nxt)

        if not nx.is_directed_acyclic_graph(G):
            return False, "Graph contains cycles."
        return True, "Graph is valid."
    except Exception as e:
        return False, f"Validation failed: {e}"

# --- API Graph Manager (Consolidated into LangGraphOpenApiRouter) ---
class ApiGraphManager:  # Keeping for potential future use or if it's used elsewhere
    llm: Optional[AzureChatOpenAI] = None
    openapi_yaml: Optional[str] = None
    _current_graph_state: Optional[Api_execution_graph] = None

    @classmethod
    def set_llm(cls, llm: AzureChatOpenAI, openapi_yaml: str):
        """Initialize the LLM instance and spec."""
        cls.llm = llm
        cls.openapi_yaml = openapi_yaml
        logging.info(f"LLM set to {cls.llm}")
        cls._current_graph_state = Api_execution_graph(graph=Graph(nodes={}))

    @classmethod
    def current_graph(cls) -> Dict:
        """Return the current graph as a dictionary."""
        return cls._current_graph_state.graph.model_dump().get("nodes", {}) if cls._current_graph_state and cls._current_graph_state.graph else {}

    @classmethod
    def update_graph(cls, graph: Dict):
        """Update the current graph state."""
        if cls._current_graph_state and cls._current_graph_state.graph:
            cls._current_graph_state.graph.nodes = {k: Node(**v) for k, v in graph.items()}
        else:
            cls._current_graph_state = Api_execution_graph(graph=Graph(nodes={k: Node(**v) for k, v in graph.items()}))

    @staticmethod
    def extract_src_dest(user_input: str) -> tuple[str, str]:
        logging.warning("Implement extract_src_dest function based on your needs.")
        return "source_node_placeholder", "destination_node_placeholder"

    @staticmethod
    def extract_node_name(user_input: str) -> str:
        logging.warning("Implement extract_node_name function based on your needs.")
        return "node_to_remove_placeholder"

    @staticmethod
    def extract_rename_mapping(user_input: str) -> tuple[str, str]:
        logging.warning("Implement extract_rename_mapping function based on your needs.")
        return "old_node_placeholder", "new_node_placeholder"

class LangGraphOpenApiRouter:
    def __init__(self, spec_text: str, llm_router: AzureChatOpenAI, llm_worker: AzureChatOpenAI, cache_dir: str = "./cache"):
        os.makedirs(cache_dir, exist_ok=True)
        ApiGraphManager.set_llm(llm_worker, spec_text)
        self.llm_router = llm_router
        self.llm_worker = llm_worker
        self.cache_dir = cache_dir
        self.tools: Dict[str, Any] = {}
        self._register_tools()
        logger.info("Available tools: %s", list(self.tools.keys()))

    def tool(self, name: str):
        def deco(fn):
            self.tools[name] = fn
            logger.info("Registered tool: %s", name)
            return fn
        return deco

    def _register_tools(self):
        @self.tool("parse_openapi")
        def parse_openapi(state: RouterState) -> RouterState:
            logger.info("Node ▶ parse_openapi; user_input=%r", state.user_input)
            regenerate = False
            if os.path.exists(schema_path):
                try:
                    state.openapi_schema = json.load(open(schema_path))
                    logger.info("Loaded schema from cache.")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Error loading cached schema (%s). Regenerating.", e)
                    regenerate = True
            else:
                regenerate = True

            if regenerate:
                logger.info("Regenerating schema/graph representation...")
                out = self._generate_api_execution_graph_fn("")
                if out and out.graph:
                    state.openapi_schema = out.graph.model_dump()
                    try:
                        with open(schema_path, "w") as f:
                            json.dump(state.openapi_schema, f)
                        logger.info("Saved regenerated schema to cache.")
                    except IOError as e:
                        logger.error("Error saving schema to cache: %s", e)
                else:
                    logger.error("Failed to regenerate schema/graph.")
                    state.openapi_schema = None
            return state

        @self.tool("generate_sequence")
        def generate_sequence(state: RouterState) -> RouterState:
            logger.info("Node ▶ generate_sequence")
            out = self._generate_api_execution_graph_fn(state.user_input or "")
            state.execution_graph = out.graph
            return state

        @self.tool("generate_payloads")
        def generate_payloads(state: RouterState) -> RouterState:
            logger.info("Node ▶ generate_payloads")
            force = "realistic" in (state.user_input or "").lower()
            regenerate = False
            if os.path.exists(payload_path) and not force:
                try:
                    state.payloads = json.load(open(payload_path))
                    logger.info("Loaded payloads from cache.")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Error loading cached payloads (%s). Regenerating.", e)
                    regenerate = True
            else:
                regenerate = True

            if regenerate:
                logger.info("Regenerating payloads...")
                payload_request = state.user_input or "generic examples"
                out = self._generate_payload_fn(payload_request)
                try:
                    payload_dict = json.loads(out.textContent)
                    state.payloads = payload_dict
                    try:
                        with open(payload_path, "w") as f:
                            json.dump(state.payloads, f, indent=2)
                        logger.info("Saved regenerated payloads to cache.")
                    except IOError as e:
                        logger.error("Error saving payloads to cache: %s", e)
                except json.JSONDecodeError:
                    logger.error("Failed to parse generated payload JSON: %s", out.textContent)
                    state.payloads = {"error": "Failed to generate valid payload JSON", "raw_output": out.textContent}
                except AttributeError:
                    logger.error("Payload generation function did not return expected TextContent object.")
                    state.payloads = {"error": "Payload generation failed internally."}

            return state

        @self.tool("answer_openapi")
        def answer_openapi(state: RouterState) -> RouterState:
            logger.info("Node ▶ answer_openapi")
            out = self._openapi_help_fn(state.user_input or "")
            state.response = out.textContent
            return state

        @self.tool("simulate_load_test")
        def simulate_load_test(state: RouterState) -> RouterState:
            logger.info("Node ▶ simulate_load_test")
            n = next((int(w) for w in (state.user_input or "").split() if w.isdigit()), 1)
            out = self._simulate_load_test_fn(num_users=n)
            state.response = out.textContent
            return state

        @self.tool("execute_workflow")
        def execute_workflow(state: RouterState) -> RouterState:
            logger.info("Node ▶ execute_workflow")
            out = self._execute_workflow_fn(state.user_input or "")
            state.response = out.textContent
            return state

        @self.tool("validate_graph")
        def validate_graph(state: RouterState) -> RouterState:
            logger.info("Node ▶ validate_graph")
            is_valid, reason = validate_graph(ApiGraphManager.current_graph())
            state.response = f"✅ Execution graph is valid." if is_valid else f"⚠️ Invalid graph: {reason}"
            return state

        @self.tool("get_execution_graph_json")
        def get_execution_graph_json(state: RouterState) -> RouterState:
            logger.info("Node ▶ get_execution_graph_json")
            state.response = json.dumps(ApiGraphManager.current_graph(), indent=2)
            return state

        @self.tool("unknown_intent")
        def unknown_intent(state: RouterState) -> RouterState:
            logger.info("Node ▶ unknown_intent")
            state.response = "Sorry, I didn't understand that or cannot perform that action."
            return state

        @self.tool("add_edge")
        def add_edge(state: RouterState) -> RouterState:
            logger.info("Node ▶ add_edge")
            try:
                prompt = f"""
                    You are an API graph assistant. A user wants to add an edge between two API nodes.

                    - Input: {state.user_input}
                    - Ensure each node has a unique name (operationId).
                    - If the destination node is reused (e.g., verifyProduct), generate a new one (e.g., verifyAfterDelete).
                    - Output: JSON like {{ "source": "createProduct", "dest": "verifyAfterDelete" }}
                    """
                llm_response = self.llm_worker([HumanMessage(content=prompt)])
                try:
                    mapping = json.loads(llm_response.content)
                    src, dest = mapping["source"], mapping["dest"]
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse LLM response for edge addition: {llm_response.content}")

                graph = ApiGraphManager.current_graph()
                if dest not in graph:
                    template = mapping.get("template_node", dest.split("After")[0] if "After" in dest else dest)
                    if template in graph:
                        graph[dest] = graph[template].copy()
                        graph[dest]["operationId"] = dest
                        if "next" not in graph[dest]:
                            graph[dest]["next"] = []
                    else:
                        raise ValueError(f"Could not find template node '{template}' in the graph.")

                updated_graph = GraphOps.add_edge(graph, src, dest)
                ApiGraphManager.update_graph(updated_graph)
                state.execution_graph = GraphOutput(graph=Graph(**{"nodes": updated_graph})) # Update graph state
                state.response = f"✅ Added edge: {src} → {dest}"
            except Exception as e:
                logger.error("Error in add_edge: %s", e)
                state.response = f"Error adding edge: {e}"
            state.next_step = None
            return state

        @self.tool("remove_node")
        def remove_node(state: RouterState) -> RouterState:
            logger.info("Node ▶ remove_node")
            try:
                prompt = f"""
                    A user wants to remove a node from the API graph.
                    - Input: {state.user_input}
                    - Extract the node name (operationId) to remove.
                    - Output the node name as a string.
                    """
                llm_response = self.llm_worker([HumanMessage(content=prompt)])
                node_to_remove = llm_response.content.strip().replace('"', '') # Extract node name

                graph = ApiGraphManager.current_graph()
                if node_to_remove not in graph:
                    raise ValueError(f"Node '{node_to_remove}' not found in the graph.")
                updated_graph = GraphOps.remove_node(graph, node_to_remove)
                ApiGraphManager.update_graph(updated_graph)
                state.execution_graph = GraphOutput(graph=Graph(**{"nodes": updated_graph}))
                state.response = f"✅ Removed node: {node_to_remove}"
            except Exception as e:
                logger.error("Error in remove_node: %s", e)
                state.response = f"Error removing node: {e}"
            state.next_step = None
            return state

        @self.tool("remove_edge")
        def remove_edge(state: RouterState) -> RouterState:
            logger.info("Node ▶ remove_edge")
            try:
                prompt = f"""
                    A user wants to remove a connection (edge) between two API nodes.

                    - Input: {state.user_input}
                    - Extract source and destination node names (operationIds)
                    - Output JSON: {{ "source": "createUser", "dest": "verifyUser" }}
                    """
                llm_response = self.llm_worker([HumanMessage(content=prompt)])
                try:
                    edge = json.loads(llm_response.content)
                    src, dest = edge["source"], edge["dest"]
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse LLM response for edge removal: {llm_response.content}")

                graph = ApiGraphManager.current_graph()
                updated_graph = GraphOps.remove_edge(graph, src, dest)
                ApiGraphManager.update_graph(updated_graph)
                state.execution_graph = GraphOutput(graph=Graph(**{"nodes": updated_graph}))
                state.response = f"❌ Removed edge: {src} → {dest}"
            except Exception as e:
                logger.error("Error in remove_edge: %s", e)
                state.response = f"Error removing edge: {e}"
            state.next_step = None
            return state

        @self.tool("rename_node")
        def rename_node(state: RouterState) -> RouterState:
            logger.info("Node ▶ rename_node")
            try:
                prompt = f"""
                    A user wants to rename an API node in the graph.

                    - Input: {state.user_input}
                    - Extract original and new node names (operationIds)
                    - Output JSON: {{ "from": "verifyUser", "to": "verifyAfterDelete" }}
                    """
                llm_response = self.llm_worker([HumanMessage(content=prompt)])
                try:
                    rename = json.loads(llm_response.content)
                    old_name, new_name = rename["from"], rename["to"]
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse LLM response for node renaming: {llm_response.content}")

                graph = ApiGraphManager.current_graph()
                if old_name not in graph:
                    raise ValueError(f"Node '{old_name}' not found in the graph.")
                if new_name in graph:
                    raise ValueError(f"Node '{new_name}' already exists.")

                graph[new_name] = graph.pop(old_name)
                graph[new_name]["operationId"] = new_name
                updated_graph = GraphOps.rename_node_in_edges(graph, old_name, new_name)
                ApiGraphManager.update_graph(updated_graph)
                state.execution_graph = GraphOutput(graph=Graph(**{"nodes": updated_graph}))
                state.response = f"🔁 Renamed '{old_name}' → '{new_name}'"
            except Exception as e:
                logger.error("Error in rename_node: %s", e)
                state.response = f"Error renaming node: {e}"
            state.next_step = None
            return state

        @self.tool("regenerate_graph")
        def regenerate_graph(state: RouterState) -> RouterState:
            logger.info("Node ▶ regenerate_graph")
            try:
                out = self._generate_api_execution_graph_fn(state.user_input or "")
                state.execution_graph = out.graph
                state.response = "🔄 Regenerated the API execution graph."
            except Exception as e:
                logger.error("Error in regenerate_graph: %s", e)
                state.response = f"Error regenerating graph: {e}"
            state.next_step = None
            return state

        @self.tool("execute_plan")
        def execute_plan(state: RouterState) -> RouterState:
            logger.info("Node ▶ execute_plan; plan=%s, next_step=%s", state.plan, state.next_step)
            current_plan = list(state.plan or [])

            if current_plan:
                step_to_execute = current_plan.pop(0)
                state.plan = current_plan
                if step_to_execute in self.tools:
                    logger.info("✅ Executing step from plan: %s", step_to_execute)
                    state = self.tools[step_to_execute](state)
                    if current_plan:
                        state.next_step = "execute_plan"
                    else:
                        logger.info("Plan finished.")
                        state.next_step = None
                else:
                    logger.warning("⚠ Unknown step in plan: %s. Skipping.", step_to_execute)
                    state.response = f"Warning: Skipped unknown plan step '{step_to_execute}'."
                    if current_plan:
                        state.next_step = "execute_plan"
                    else:
                        state.next_step = None
            elif state.next_step and state.next_step != "execute_plan":
                step = state.next_step
                state.next_step = None
                state.plan = None
                if step in self.tools:
                    logger.info("✅ Executing single next step: %s", step)
                    state = self.tools[step](state)
                    state.next_step = None
                else:
                    logger.warning("⚠ Unknown next step: %s. Routing to 'unknown_intent'.", step)
                    state.next_step = "unknown_intent"
            else:
                logger.info("ℹ️ No plan or next_step to execute.")
                state.next_step = None

            return state

    def route_intent(self, state: RouterState) -> RouterState:
        logger.info("Node ▶ route_intent; user_input=%r, last_response=%r", state.user_input, state.response)
        if state.next_step != "execute_plan":
            state.next_step = None
            state.plan = None

        # Basic attempt to avoid redundancy: Check if the same query was asked recently
        if state.user_input and state.user_input in [item[0] for item in state.action_history[-2:]]: # Check last 2 inputs
            state.response = "It seems you asked this recently. How about trying something new?"
            state.next_step = None # Go back to router
            return state

        prompt = (
            "You are an OpenAPI assistant router.\n"
            f"The user said: '{state.user_input}'\n\n"
            "Current graph/API state summary:\n"
            f"- Payloads generated: {'yes' if state.payloads else 'no'}\n"
            f"- Execution graph available: {'yes' if state.execution_graph else 'no'}\n"
            f"Available tools: {list(self.tools.keys())}\n"
            "Consider the user's goal. If the user asks a question, use 'answer_openapi'. If they want to run something, use 'execute_workflow' or 'simulate_load_test'. If they want to build/modify a sequence, use 'generate_sequence' or graph tools ('add_edge', etc.). If they ask to generate payloads, use 'generate_payloads'."
            "If a multi-step process is needed, you can optionally output a plan.\n"
            "Output your choice as JSON: {\"next_step\": \"tool_name\"} or {\"plan\": [\"tool1\", \"tool2\", ...]}"
        )

        response = self.llm_router([HumanMessage(content=prompt)])
        next_action = response.content.strip()
        logger.info("Router LLM response: %s", next_action)

        try:
            output = json.loads(next_action)
            if "plan" in output and isinstance(output["plan"], list) and output["plan"]:
                state.plan = output["plan"]
                state.next_step = "execute_plan"
                logger.info("Routing to execute_plan with plan: %s", state.plan)
            elif "next_step" in output and output["next_step"] in self.tools:
                state.next_step = output["next_step"]
                state.plan = None
                logger.info("Routing to next_step: %s", state.next_step)
            else:
                logger.warning("Router LLM provided invalid JSON or unknown tool. Falling back.")
                state.next_step = "unknown_intent"
                state.plan = None

        except json.JSONDecodeError:
            if next_action in self.tools:
                state.next_step = next_action
                state.plan = None
                logger.info("Routing to next_step (non-JSON): %s", state.next_step)
            else:
                logger.warning("Router LLM response is not a known tool or valid JSON. Falling back.")
                state.next_step = "unknown_intent"
                state.plan = None

        if state.user_input:
            state.action_history.append((state.user_input, state.next_step)) # Record the action
            if len(state.action_history) > 10: # Keep a limited history
                state.action_history.pop(0)

        return state

    def should_continue(self, state: RouterState) -> str:
        """Determines the next node after a tool runs."""
        logger.debug("Evaluating 'should_continue': next_step=%s, plan=%s, response=%s",
                     state.next_step, state.plan, state.response)

        if state.next_step == "execute_plan" and state.plan:
            logger.info("Conditional edge: Continuing plan -> execute_plan")
            return "execute_plan"
        elif state.next_step and state.next_step != "execute_plan":
            logger.info("Conditional edge: Explicit next_step -> %s", state.next_step)
            return state.next_step
        else:
            logger.info("Conditional edge: Defaulting back to -> route_intent")
            return "route_intent"

    def build_graph(self):
        builder = StateGraph(RouterState)

        builder.add_node("route_intent", self.route_intent)
        for name, fn in self.tools.items():
            logger.info("ADDING NODE: %s", name)
            builder.add_node(name, fn)

        builder.set_entry_point("route_intent")

        builder.add_conditional_edges(
            "route_intent",
            lambda s: s.next_step if s.next_step in self.tools else "unknown_intent",
            {k: k for k in list(self.tools.keys()) + ["unknown_intent"]}
        )

        for name in self.tools.keys():
            builder.add_conditional_edges(
                name,
                self.should_continue,
                {"route_intent": "route_intent",
                 "execute_plan": "execute_plan",
                 **{k: k for k in self.tools.keys()}}
            )

        graph = builder.compile()
        logger.info("COMPILED NODES: %s", graph.nodes)

        return graph

    # --- Consolidated Tool Methods from ApiGraphManager ---
    def _generate_api_execution_graph_fn(self, user: str) -> Api_execution_graph:
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
        llm_response = call_llm(
            formatted, self.llm_worker, Api_execution_graph
        )
        ApiGraphManager._current_graph_state = llm_response
        return ApiGraphManager._current_graph_state

    def _generate_payload_fn(self, user: str) -> TextContent:
        """Generate a realistic JSON payload matching the given OpenAPI schema."""
        logging.info("Starting generate_payload_fn")
        prompt = f"Given the OpenAPI specification, generate a JSON payload for: {user}"
        llm_response = call_llm(
            prompt, self.llm_worker, TextContent
        )
        return llm_response

    def _openapi_help_fn(self, user: str) -> TextContent:
        """Explain OpenAPI endpoints. Answers user query for given OpenAPI swagger."""
        logging.info("Starting openapi_help_fn")
        prompt = f"Based on the OpenAPI specification, answer user query: {user}"
        llm_response = call_llm(
            prompt, self.llm_worker, TextContent
        )
        return llm_response

    def _simulate_load_test_fn(self, num_users: int) -> TextContent:
        """Simulates a load test on the API."""
        logging.info(f"Starting simulate_load_test_fn for {num_users} users.")
        # This is a placeholder - you'd need actual logic here to simulate API calls
        return TextContent(textContent=f"Simulated load test for {num_users} users.")

    def _execute_workflow_fn(self, user_input: str) -> TextContent:
        """Executes a predefined workflow based on user input."""
        logging.info(f"Starting execute_workflow_fn with input: {user_input}")
        # This is a placeholder - you'd need logic to map user input to specific workflows
        return TextContent(textContent=f"Executed workflow for: {user_input}")


if __name__ == "__main__":
    spec_file = "petstore.yaml" # Or path to your spec
    try:
        with open(spec_file, 'r') as f:
            spec = f.read()
    except FileNotFoundError:
        logger.error(f"Error: OpenAPI spec file not found at '{spec_file}'")
        exit(1)
    except IOError as e:
        logger.error(f"Error reading spec file '{spec_file}': {e}")
        exit(1)

    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    if not all(os.getenv(var) for var in required_env_vars):
        logger.error("Missing required Azure OpenAI environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)")
        exit(1)

    try:
        router_llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_deployment="gpt-35-router", # Replace with your deployment name
            temperature=0,
            max_tokens=500
        )
        worker_llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_deployment="gpt-35-worker", # Replace with your deployment name
            temperature=0,
            max_tokens=1500
        )
        logger.info("Azure OpenAI LLMs initialized.")

    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI models: {e}")
        exit(1)

    router = LangGraphOpenApiRouter(spec, router_llm, worker_llm)
    graph = router.build_graph()
    logger.info("Graph built successfully.")

    print("Assistant: Hello! How can I help you with the OpenAPI spec today?")
    while True:
        try:
            ui = input("You: ")
            if ui.lower() in ["exit", "quit", "bye"]:
                break
            final_state = graph.invoke({"user_input": ui})

            assistant_response = "Could not determine response."
            if isinstance(final_state, dict) and 'response' in final_state:
                assistant_response = final_state['response']
            elif isinstance(final_state, RouterState):
                assistant_response = final_state.response

            if not assistant_response:
                assistant_response = "OK. What next?"

            print(f"Assistant: {assistant_response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error("An error occurred during processing: %s", e, exc_info=True)
            print("Assistant: Sorry, an internal error occurred.")
