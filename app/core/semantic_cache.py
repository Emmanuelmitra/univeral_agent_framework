"""Layer 5: Semantic Cache with Cosine Similarity Thresholding & CDC Invalidation."""

import math
import re
import time
from collections import Counter
from typing import Any


class SemanticCacheEngine:
    """Caches static/FAQ responses using cosine similarity and supports CDC invalidation."""

    def __init__(self, similarity_threshold: float = 0.88):
        self.similarity_threshold = similarity_threshold
        self._store: list[dict[str, Any]] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _tokenize(text: str) -> Counter:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return Counter(words)

    @classmethod
    def _cosine_similarity(cls, vec1: Counter, vec2: Counter) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        sum1 = sum(v ** 2 for v in vec1.values())
        sum2 = sum(v ** 2 for v in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        return numerator / denominator if denominator else 0.0

    def is_dynamic_query(self, query: str) -> bool:
        """Selective Caching: Bypass cache for personalized, order-specific, or live-mutation queries."""
        dynamic_markers = ["ord-", "inv-", "refund", "credit", "now", "current time", "calculate", "my account"]
        q_low = query.lower()
        return any(m in q_low for m in dynamic_markers)

    def lookup(self, tenant_id: str, query: str) -> dict[str, Any] | None:
        if self.is_dynamic_query(query):
            self.misses += 1
            return None

        q_vec = self._tokenize(query)
        best_score = 0.0
        best_entry = None

        for entry in self._store:
            if entry["tenant_id"] != tenant_id:
                continue
            score = self._cosine_similarity(q_vec, entry["vector"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            self.hits += 1
            return {
                "cached_response": best_entry["response"],
                "similarity_score": round(best_score, 4),
                "cached_at": best_entry["timestamp"],
            }

        self.misses += 1
        return None

    def store(self, tenant_id: str, query: str, response: str, tags: list[str] | None = None) -> None:
        if self.is_dynamic_query(query):
            return
        self._store.append({
            "tenant_id": tenant_id,
            "query": query,
            "vector": self._tokenize(query),
            "response": response,
            "tags": tags or [],
            "timestamp": time.time(),
        })

    def invalidate_cdc(self, tenant_id: str | None = None, keyword: str | None = None) -> int:
        """Change Data Capture (CDC) webhook handler: evicts stale cache entries when docs update."""
        before = len(self._store)
        if tenant_id is None and keyword is None:
            self._store.clear()
            return before

        kept = []
        for item in self._store:
            match_tenant = tenant_id and item["tenant_id"] == tenant_id
            match_kw = keyword and (keyword.lower() in item["query"].lower() or keyword.lower() in item["response"].lower())
            if match_tenant or match_kw:
                continue
            kept.append(item)
        self._store = kept
        return before - len(self._store)
