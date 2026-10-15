"""Layer 6: Minimum Sufficient Intelligence (MSI) Progressive Complexity Router."""

import re


class MinimumSufficientIntelligenceRouter:
    """Routes queries along the cost/intelligence ladder:
    Tier 1: Deterministic Regex/Heuristics (0 LLM cost, <2ms)
    Tier 2: Fast Flash LLM (`gemini-2.5-flash`) for standard tool & RAG tasks
    Tier 3: Frontier Reasoning LLM (`gemini-2.5-pro`) for multi-step analytics, math, or root-cause analysis
    """

    GREETING_REGEX = re.compile(r"^\s*(hi|hello|hey|good morning|good evening|ping|status)\s*[!?.]*\s*$", re.I)

    def route(self, query: str, cartridge_rules: dict[str, str] | None = None) -> dict:
        cleaned = query.strip()

        # 1. Check cartridge deterministic quick-rules first
        if cartridge_rules:
            for pattern, static_reply in cartridge_rules.items():
                if re.search(pattern, cleaned, re.IGNORECASE):
                    return {
                        "tier": "TIER_1_DETERMINISTIC_RULE",
                        "model": "deterministic_engine",
                        "direct_response": static_reply,
                    }

        # 2. Standard greeting / health ping shortcut
        if self.GREETING_REGEX.match(cleaned):
            return {
                "tier": "TIER_1_DETERMINISTIC_HEURISTIC",
                "model": "deterministic_engine",
                "direct_response": "Hello! I am online and ready. How can I help you today?",
            }

        # 3. Complexity scoring for Tier 2 (Flash) vs Tier 3 (Pro)
        complex_keywords = [
            "root cause", "forecast", "anomaly", "optimize", "compare", "architecture",
            "multi-step", "regression", "backtest", "audit", "why did", "dispute",
        ]
        score = sum(1 for kw in complex_keywords if kw in cleaned.lower())
        if len(cleaned) > 350 or score >= 2:
            return {
                "tier": "TIER_3_FRONTIER_REASONING",
                "model": "gemini-2.5-pro",
                "direct_response": None,
            }

        return {
            "tier": "TIER_2_STANDARD_FLASH",
            "model": "gemini-2.5-flash",
            "direct_response": None,
        }
