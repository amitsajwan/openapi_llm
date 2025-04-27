# trustcall.py
from typing import Any, Callable, Tuple

class TrustResult:
    def __init__(self, success: bool, output: Any = None, error: str = ""):
        self.success = success
        self.output = output
        self.error = error

def trustcall(func: Callable, *args, **kwargs) -> TrustResult:
    """
    Calls func(*args, **kwargs).  
    Returns TrustResult(success, output, error).
    """
    try:
        out = func(*args, **kwargs)
        return TrustResult(success=True, output=out)
    except Exception as e:
        return TrustResult(success=False, error=str(e))
