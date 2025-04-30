from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RouterState:
    user_input: Optional[str] = None
    openapi_schema: Optional[Any] = None
    execution_graph: Optional[Dict[str, Any]] = None
    payloads: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    intent: Optional[str] = None
    next_step: Optional[str] = None
    plan: List[str] = field(default_factory=list)
    action_history: List[tuple] = field(default_factory=list)
    scratchpad: Dict[str, Any] = field(default_factory=dict)  # persistent memory

