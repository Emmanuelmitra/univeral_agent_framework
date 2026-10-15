"""Layer 2 & 3: Token-Bucket Rate Limiter, Tenant Isolation & Daily Token Quota Enforcer."""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TenantBucket:
    tokens: float = 30.0
    last_refill: float = field(default_factory=time.time)
    daily_tokens_used: int = 0


class EnterpriseTrafficController:
    """Enforces per-tenant request rate limits (req/min) and daily LLM token budgets."""

    def __init__(self, rate_per_minute: int = 30, daily_token_cap: int = 250_000):
        self.rate_per_minute = rate_per_minute
        self.daily_token_cap = daily_token_cap
        self._buckets: dict[str, TenantBucket] = defaultdict(TenantBucket)

    def check_and_consume(self, tenant_id: str, user_id: str, estimated_tokens: int = 250) -> dict:
        key = f"{tenant_id}:{user_id}"
        bucket = self._buckets[key]
        now = time.time()

        # Refill token bucket smoothly
        elapsed = now - bucket.last_refill
        refill = elapsed * (self.rate_per_minute / 60.0)
        bucket.tokens = min(float(self.rate_per_minute), bucket.tokens + refill)
        bucket.last_refill = now

        if bucket.tokens < 1.0:
            return {
                "allowed": False,
                "reason": f"RATE_LIMIT_EXCEEDED: Max {self.rate_per_minute} requests/minute for {key}.",
            }

        if bucket.daily_tokens_used + estimated_tokens > self.daily_token_cap:
            return {
                "allowed": False,
                "reason": f"TOKEN_BUDGET_EXCEEDED: Daily cap of {self.daily_token_cap} tokens reached for {key}.",
            }

        bucket.tokens -= 1.0
        bucket.daily_tokens_used += estimated_tokens
        return {
            "allowed": True,
            "remaining_rpm_tokens": round(bucket.tokens, 2),
            "daily_tokens_used": bucket.daily_tokens_used,
        }
