"""Layer 4, 9 & 15: 360-Degree Input/Tool/Output Guardrails + Cyclic Tool Loop Breaker."""

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any
from google.genai import types

logger = logging.getLogger("universal_agent.guardrails")

DEFAULT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)you\s+are\s+now\s+in\s+DAN\s+mode",
    r"(?i)system\s+override\s+code",
    r"(?i)reveal\s+your\s+hidden\s+system\s+prompt",
    r"(?i)disregard\s+all\s+rules\s+and\s+guardrails",
]

DEFAULT_BLOCKED_TOOL_ARGS = [
    "/etc/passwd", "/etc/shadow", "rm -rf", "DROP TABLE",
    "DROP DATABASE", "TRUNCATE", ";--", "UNION SELECT",
]

DEFAULT_SENSITIVE_OUTPUT_PATTERNS = [
    r"\b[A-Za-z0-9_-]{39}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d{4}[ -]?){3}\d{4}\b",
]


class EnterpriseGuardrailEngine:
    """Enforces Input, Tool, Loop-Breaker, and Output DLP Guardrails (ADK 2.8+ compatible)."""

    def __init__(self, cartridge_module: Any):
        extra_inputs = getattr(cartridge_module, "EXTRA_BLOCKED_INPUT_PATTERNS", [])
        extra_tool_args = getattr(cartridge_module, "EXTRA_BLOCKED_TOOL_ARGS", [])
        extra_outputs = getattr(cartridge_module, "EXTRA_SENSITIVE_OUTPUT_PATTERNS", [])
        self.max_input_length = getattr(cartridge_module, "MAX_INPUT_LENGTH", 6000)

        self.input_regexes = [
            re.compile(p, re.IGNORECASE) for p in (DEFAULT_INJECTION_PATTERNS + extra_inputs)
        ]
        self.blocked_tool_args = [arg.lower() for arg in (DEFAULT_BLOCKED_TOOL_ARGS + extra_tool_args)]
        self.output_regexes = [
            re.compile(p) for p in (DEFAULT_SENSITIVE_OUTPUT_PATTERNS + extra_outputs)
        ]
        self._tool_call_counts: dict[str, int] = defaultdict(int)

    def scrub_text(self, text: str) -> str:
        cleaned = text
        for regex in self.output_regexes:
            cleaned = regex.sub("[REDACTED_BY_DLP_GUARDRAIL]", cleaned)
        return cleaned

    def input_guardrail(self, callback_context: Any = None, **kwargs) -> types.Content | None:
        ctx = callback_context or kwargs.get("ctx")
        if ctx is None:
            return None

        user_text = ""
        user_content = getattr(ctx, "user_content", None)
        if user_content and getattr(user_content, "parts", None):
            for part in user_content.parts:
                if hasattr(part, "text") and part.text:
                    user_text += part.text

        if not user_text:
            return None

        if len(user_text) > self.max_input_length:
            return types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text=f"[GUARDRAIL BLOCKED] Input length ({len(user_text)}) exceeds limit ({self.max_input_length})."
                )]
            )

        for regex in self.input_regexes:
            if regex.search(user_text):
                return types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="[GUARDRAIL VIOLATION] Request blocked by enterprise security policy (prompt injection or restricted override detected)."
                    )]
                )
        return None

    def before_tool_guardrail(
        self, tool: Any = None, args: dict[str, Any] | None = None, tool_context: Any = None, **kwargs
    ) -> dict[str, Any] | None:
        tool_obj = tool or kwargs.get("tool")
        tool_args = args if args is not None else kwargs.get("args", {})
        tool_name = getattr(tool_obj, "name", str(tool_obj))
        arg_str = str(tool_args).lower()

        for bad_pattern in self.blocked_tool_args:
            if bad_pattern in arg_str:
                return {
                    "status": "BLOCKED_BY_TOOL_GUARDRAIL",
                    "reason": f"Argument contains prohibited pattern '{bad_pattern}'.",
                }

        fingerprint = hashlib.sha256(f"{tool_name}:{arg_str}".encode()).hexdigest()[:16]
        self._tool_call_counts[fingerprint] += 1
        if self._tool_call_counts[fingerprint] > 3:
            return {
                "status": "LOOP_CIRCUIT_BREAKER_TRIPPED",
                "reason": f"Tool '{tool_name}' was invoked >3 times with identical arguments. Halting loop; synthesize final answer now.",
            }
        return None

    def after_tool_guardrail(
        self, tool: Any = None, args: dict[str, Any] | None = None, tool_context: Any = None, tool_response: Any = None, **kwargs
    ) -> Any:
        resp = tool_response if tool_response is not None else kwargs.get("response")
        if resp is None:
            return None
        raw_str = str(resp)
        scrubbed = self.scrub_text(raw_str)
        if scrubbed != raw_str:
            return {"sanitized_tool_output": scrubbed, "dlp_redacted": True}
        return resp
