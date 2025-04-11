from typing import Annotated, Dict, List, Any
from langgraph.graph import StateGraph

state_schema = {
    "responses": Annotated[List[Dict[str, Any]], "merge"]
}
graph = StateGraph(state_schema=state_schema)




return {"responses": [{operation_id: result}]}
