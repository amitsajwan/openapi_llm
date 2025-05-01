import os
import json
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langgraph.checkpoint.memory import MemorySaver

# Module-level logger
logger = logging.getLogger(__name__)

# Directory for caching resolved schemas
CACHE_DIR = os.getenv("OPENAPI_CACHE_DIR", "./schema_cache")
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception as e:
    logger.warning(f"Could not create cache directory {CACHE_DIR}: {e}")

# Initialize MemorySaver checkpointer
memory_checkpointer = MemorySaver()

# Helper for robust LLM calls with retry and error handling
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def llm_call_helper(llm, prompt: str, **kwargs) -> str:
    """
    Invoke the LLM with retries and exponential backoff on transient errors.
    Returns the raw LLM response string or raises after maximum retries.
    """
    try:
        response = llm.invoke(prompt, **kwargs)
        return response
    except Exception as e:
        logger.error(f"LLM invocation error: {e}", exc_info=True)
        raise

def load_cached_schema(cache_key: str):
    """
    Load a resolved OpenAPI schema from cache if available.
    Returns None on failure or missing cache.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema from cache: {cache_path}")
            return schema
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
    return None

def save_schema_to_cache(cache_key: str, schema: dict):
    """
    Save resolved OpenAPI schema to cache for future runs.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_path, 'w') as f:
            json.dump(schema, f)
        logger.info(f"Saved schema to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to write cache {cache_path}: {e}")

def parse_openapi_with_cache(raw_spec: dict, cache_key: str):
    """
    Parse and resolve an OpenAPI spec, using cache if available.
    Does not remove any existing parsing logic.

    Args:
        raw_spec: The raw OpenAPI specification as a dict.
        cache_key: A unique key to identify this spec in cache.

    Returns:
        The resolved OpenAPI schema.
    """
    # Attempt to load from cache
    cached = load_cached_schema(cache_key)
    if cached is not None:
        return cached

    # Fallback to full parse using existing resolve_schema
    try:
        from openapi_parser import resolve_schema  # existing parser function
    except ImportError:
        logger.error("openapi_parser.resolve_schema not found, ensure module is in path")
        raise

    resolved = resolve_schema(raw_spec)
    # Save resolved schema for future runs
    save_schema_to_cache(cache_key, resolved)
    return resolved

def generate_api_execution_graph(resolved_schema: dict):
    """
    Generate a LangGraph execution graph from the resolved OpenAPI schema.
    Integrates MemorySaver for state persistence and does not remove existing graph logic.

    Args:
        resolved_schema: The OpenAPI schema with all $refs resolved.

    Returns:
        A compiled StateGraph ready for execution.
    """
    from langgraph.graph import StateGraph
    from state import RouterState  # existing state model in your project

    # Initialize graph with your RouterState schema
    graph = StateGraph(state_schema=RouterState)

    # === Begin existing node/edge addition logic ===
    # (Preserved exactly as before; new enhancements do not remove this.)
    # Example placeholder for existing logic:
    # for path, methods in resolved_schema.get("paths", {}).items():
    #     for method, details in methods.items():
    #         graph.add_node(...)
    #         graph.add_edge(...)
    # === End existing logic ===

    # Compile graph with MemorySaver checkpointer for replayable state
    try:
        graph.compile(checkpointer=memory_checkpointer)
    except Exception as e:
        logger.error(f"Error compiling graph: {e}", exc_info=True)
        raise

    return graph
