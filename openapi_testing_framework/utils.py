"""
utils.py

General helper functions:
  - llm_call_helper: wraps LLM calls with retries, structured function-calling, JSON parsing
  - read_swagger_files: list OpenAPI YAML/JSON files in a folder
"""
import json
import os
import logging
from typing import Any, Dict, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_swagger_files(folder: str) -> list:
    """
    Return a list of file paths for .yaml, .yml, or .json files in the given folder.
    """
    files = []
    for fname in os.listdir(folder):
        if fname.endswith(('.yaml', '.yml', '.json')):
            files.append(os.path.join(folder, fname))
    return files


def _parse_json_response(response_str: str) -> Any:
    """
    Safely parse a JSON string returned by LLM; logs and raises on error.
    """
    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from LLM response: %s", response_str)
        raise


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def llm_call_helper(
    prompt: str,
    function_name: Optional[str] = None,
    retries: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Call the OpenAI LLM with exponential backoff retries.
    If function_name is provided, uses function-calling to get structured JSON.
    Returns the parsed JSON dict.

    Args:
      prompt: the text prompt for the LLM
      function_name: if provided, the name of the function schema to call
      retries: number of retry attempts (overridden by decorator)
      model: LLM model to use
      temperature: sampling temperature

    Returns:
      Dict parsed from the LLM's JSON response
    """
    logger.info("LLM call (model=%s, function=%s) prompt: %s", model, function_name, prompt)

    # build messages
    messages = [{"role": "user", "content": prompt}]

    if function_name:
        # instruct LLM to output JSON only
        functions = [{
            "name": function_name,
            "parameters": {"type": "object"},
            "description": f"Function schema for {function_name}"
        }]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call={"name": function_name},
            temperature=temperature
        )
        content = response.choices[0].message.function_call.arguments
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        content = response.choices[0].message.content

    result = _parse_json_response(content)
    logger.info("LLM response parsed: %s", result)
    return result
