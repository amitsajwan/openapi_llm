import logging
import json
from typing import Any

from tools import llm_call_helper, update_scratchpad_reason
from models import BotState, BotIntent, AVAILABLE_TOOLS, load_state, save_state, TextContent

# Module-level logger
logger = logging.getLogger(__name__)

def init_router(router_llm: Any, session_id: str, cache_dir: str):
    """
    Initialize router with LLM, session context, and persistent cache.
    """
    return LangGraphOpenApiRouter(
        llm_router=router_llm,
        session_id=session_id,
        cache_dir=cache_dir,
    )

class LangGraphOpenApiRouter:
    def __init__(
        self, llm_router: Any, session_id: str, cache_dir: str
    ):
        self.llm_router = llm_router
        self.session_id = session_id
        self.cache_dir = cache_dir

    def route(self, state: BotState) -> BotState:
        """
        Entry point for routing a user message to the next tool.
        Reloads any persisted state, executes a routing step, then persists.
        """
        # Reload previous state if available
        persisted = load_state(self.session_id, self.cache_dir)
        if persisted:
            logger.info(f"Reloaded persisted state for session {self.session_id}")
            state = persisted

        # Determine next step
        next_step = self._route_step(state)

        # Persist updated state after routing
        try:
            save_state(state, self.session_id, self.cache_dir)
            logger.info(f"Saved state for session {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

        # Dispatch to tool
        tool_fn = AVAILABLE_TOOLS.get(next_step, None)
        if not tool_fn:
            logger.error(f"No tool function found for step '{next_step}'")
            state.text_content = TextContent(text="Internal routing error: unknown step.")
            return state

        return tool_fn(state, self.llm_router)

    def _route_step(self, state: BotState) -> str:
        """
        Core routing logic: ask the LLM which intent/tool to invoke next.
        """
        prompt = (
            f"User input: {state.user_input}\n"
            f"Current intent: {state.intent}\n"
            "Choose one of the following intents with no extra text: "
            f"{[i.value for i in BotIntent]}"
        )

        try:
            res = llm_call_helper(self.llm_router, prompt).strip()
            logger.debug(f"Router LLM response: {res}")
        except Exception as e:
            logger.error(f"Router LLM failed: {e}", exc_info=True)
            res = BotIntent.UNKNOWN.value

        if res not in [i.value for i in BotIntent]:
            logger.warning(f"Router returned invalid intent '{res}', defaulting to UNKNOWN")
            res = BotIntent.UNKNOWN.value

        state.intent = BotIntent(res)
        update_scratchpad_reason(state, "Router", f"Chose intent {res}")
        return res

    def _extract_parameters_step(self, state: BotState) -> None:
        """
        Extract parameters for the chosen intent using the LLM, with retries.
        """
        if not state.intent:
            logger.warning("_extract_parameters_step called without intent set")
            return

        if state.intent not in [BotIntent.ADD_EDGE, BotIntent.GENERATE_PAYLOAD]:
            return

        prompt = (
            f"Extract parameters as JSON for intent '{state.intent.value}'.\n"
            f"User input: {state.user_input}\n"
            "Output only the JSON object."
        )
        try:
            res = llm_call_helper(self.llm_router, prompt).strip()
            logger.debug(f"Param extraction response: {res}")
            params = json.loads(res)
            state.extracted_params = params
            update_scratchpad_reason(state, "ParamExtraction", f"Extracted params: {params}")
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            state.extracted_params = None
            update_scratchpad_reason(state, "ParamExtraction", f"Extraction error: {e}")
