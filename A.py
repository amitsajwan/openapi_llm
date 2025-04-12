from typing import TypedDict, Annotated, List, Dict, Any
import operator

class GraphState(TypedDict):
    data: Dict[str, Any]
    extracted_ids: Annotated[List[Dict[str, Any]], operator.add]  # ✅ list of dicts
    operations: Annotated[List[Dict[str, Any]], operator.add]     # ✅ list of dicts





def extract_id_node(state: GraphState) -> GraphState:
    extracted = {
        "id": 123,
        "price": 100
    }

    return {
        "extracted_ids": [extracted],
        "operations": [
            {
                "extract_id_node": {
                    "extracted": extracted,
                    "status": "ok"
                }
            }
        ]
    }
