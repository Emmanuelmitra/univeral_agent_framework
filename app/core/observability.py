"""Layer 16: Real-Time Observability, P95 Latency Tracker & Token Cost Estimator."""

from dataclasses import dataclass, field


@dataclass
class SystemMetricsStore:
    latencies_ms: list[float] = field(default_factory=list)
    total_requests: int = 0
    total_errors_recovered: int = 0
    total_tokens_estimated: int = 0
    tier_counts: dict[str, int] = field(default_factory=lambda: {
        "CACHE_HIT": 0,
        "TIER_1_DETERMINISTIC": 0,
        "TIER_2_STANDARD_FLASH": 0,
        "TIER_3_FRONTIER_REASONING": 0,
    })

    def record_request(self, latency_ms: float, tokens: int, tier: str, recovered_error: bool = False):
        self.total_requests += 1
        self.latencies_ms.append(round(latency_ms, 2))
        if len(self.latencies_ms) > 1000:
            self.latencies_ms.pop(0)
        self.total_tokens_estimated += tokens
        if recovered_error:
            self.total_errors_recovered += 1
        if tier in self.tier_counts:
            self.tier_counts[tier] += 1

    def summary(self, cache_hits: int, cache_misses: int) -> dict:
        sorted_lat = sorted(self.latencies_ms) if self.latencies_ms else [0.0]
        p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
        p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
        estimated_cost_usd = round((self.total_tokens_estimated / 1_000_000) * 0.35, 6)
        total_cache = max(cache_hits + cache_misses, 1)
        return {
            "total_requests": self.total_requests,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "semantic_cache_hit_ratio": round(cache_hits / total_cache, 3),
            "estimated_total_tokens": self.total_tokens_estimated,
            "estimated_cost_usd": estimated_cost_usd,
            "recovered_errors_count": self.total_errors_recovered,
            "routing_distribution": self.tier_counts,
        }
