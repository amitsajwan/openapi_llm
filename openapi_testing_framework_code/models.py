from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class BotState(BaseModel):
    spec_path: Optional[str] = None
    openapi_schema: Optional[Dict[str, Any]] = None
    endpoints: List[Dict[str, Any]] = []
    sequence: List[Dict[str, Any]] = []
    payloads: Dict[int, Any] = {}
    results: List[Dict[str, Any]] = []
    plan: List[Dict[str, Any]] = []
