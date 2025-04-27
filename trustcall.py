import logging
import asyncio
import tenacity
from langchain.output_parsers import JsonOutputKeyToolsParser
from langchain_core.output_parsers import JsonOutputKeyToolsParser as CoreParser

# Use core or langchain depending on version
OutputParser = JsonOutputKeyToolsParser

# Safe trustcall with retry
@tenacity.retry(
    wait=tenacity.wait_fixed(2),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
async def trustcall(prompt: str, llm, *, max_tokens: int = 1000) -> dict:
    """
    Trustable LLM call with retry and error handling.
    """
    logging.info(f"Invoking LLM with prompt length {len(prompt)}")
    response = await llm.ainvoke(
        prompt,
        config={"max_tokens": max_tokens}
    )
    logging.info("Received LLM response")
    return response

async def trustcall_safe(prompt: str, llm, *, max_tokens: int = 1000) -> dict:
    """
    Safe version of trustcall. Always returns dict (error or success).
    """
    try:
        return await trustcall(prompt, llm, max_tokens=max_tokens)
    except Exception as e:
        logging.error(f"trustcall_safe failed: {e}")
        return {"error": str(e)}

def create_extractor(schema_model):
    """
    Create an LLM output parser tied to a schema.

    Args:
        schema_model: Pydantic model to validate and parse LLM output.
    
    Returns:
        Parser instance that can parse LLM response into schema_model.
    """
    logging.info(f"Creating extractor for {schema_model.__name__}")
    parser = OutputParser(pydantic_schema=schema_model)
    return parser


from trustcall import trustcall_safe, create_extractor

def call_llm(formatted_prompt: str, llm) -> Api_execution_graph:
    # inside your method:
    
    logging.info("Calling LLM...")
    response = await trustcall_safe(formatted_prompt, llm)
    
    if "error" not in response:
        extractor = create_extractor(Api_execution_graph)
        parsed_response = extractor.parse(response)
    else:
        # handle error
        parsed_response = Api_execution_graph(error=response["error"])
