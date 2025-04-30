from typing import Any, Dict, Tuple
import networkx as nx

def validate_execution_graph(graph: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that `graph` (a dict of node -> { next: [...] }) is a DAG.
    """
    G = nx.DiGraph()
    for node, data in graph.items():
        for nxt in data.get("next", []):
            G.add_edge(node, nxt)
    if nx.is_directed_acyclic_graph(G):
        return True, "Graph is a valid DAG."
    return False, "Graph contains cycles."  # :contentReference[oaicite:8]{index=8}

def add_edge_to_graph(graph: Dict[str, Any], src: str, tgt: str) -> Dict[str, Any]:
    """
    Add an edge src->tgt to the execution graph safely.
    """
    # Ensure nodes exist
    graph.setdefault(src, {"next": []})
    graph.setdefault(tgt, {"next": []})
    if tgt not in graph[src]["next"]:
        graph[src]["next"].append(tgt)
    return graph
