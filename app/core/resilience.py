"""Layer 9 & 10: Never-Breakable Tool Execution Wrapper & Audit Logger."""

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("universal_agent.resilience")


def resilient_tool(func: Callable) -> Callable:
    """Wraps any Python function into a fault-tolerant ADK tool that never crashes in production."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict[str, Any]:
        start_time = time.perf_counter()
        tool_name = func.__name__
        try:
            result = func(*args, **kwargs)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if isinstance(result, dict):
                result["_audit"] = {"tool": tool_name, "status": "SUCCESS", "duration_ms": elapsed_ms}
                return result
            return {
                "result": result,
                "_audit": {"tool": tool_name, "status": "SUCCESS", "duration_ms": elapsed_ms},
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("Tool '%s' failed safely after %sms: %s", tool_name, elapsed_ms, exc)
            return {
                "status": "TOOL_EXECUTION_RECOVERED_ERROR",
                "tool_name": tool_name,
                "error_type": type(exc).__name__,
                "diagnostic_message": str(exc),
                "instruction_to_agent": (
                    "The tool encountered a runtime error. Do not crash; inform the user "
                    "gracefully or use an alternative reasoning path."
                ),
                "_audit": {"tool": tool_name, "status": "RECOVERED_ERROR", "duration_ms": elapsed_ms},
            }

    return wrapper
