from pydantic import BaseModel, Field
from typing import List, Optional

class Node(BaseModel):
    operationId: str
    method: str
    path: str
    payload: Optional[dict] = Field(default_factory=dict)

class Edge(BaseModel):
    from_node: str
    to_node: str
    can_parallel: bool
    reason: Optional[str] = None
    requires_human_validation: bool = False

class GraphOutput(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
