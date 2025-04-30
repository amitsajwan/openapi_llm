from typing import Any, Dict, List
from collections import defaultdict, deque

def suggest_execution_order(spec: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a simple execution graph: each operationId points to next operations.
    Orders paths in the order they appear, with no dependencies by default.
    """
    # Extract operationIds in appearance order :contentReference[oaicite:5]{index=5}
    ops = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            op_id = details.get("operationId", f"{method}_{path}")
            ops.append(op_id)

    # Naïve linear chain: op[i] -> op[i+1] :contentReference[oaicite:6]{index=6}
    graph: Dict[str, Dict[str, List[str]]] = {}
    for i, op in enumerate(ops):
        graph[op] = {"next": [ops[i+1]] if i+1 < len(ops) else []}
    return graph
