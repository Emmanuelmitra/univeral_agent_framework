"""Layer 7: Hybrid Search RAG Engine (Sparse BM25 + Dense Cosine RRF + Tenant Metadata ACLs)."""

import math
import re
from collections import Counter
from typing import Any


class HybridRagEngine:
    """Combines BM25 keyword scoring and dense token-vector similarity via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, documents: list[dict[str, Any]] | None = None):
        self.documents: list[dict[str, Any]] = documents or []

    def ingest_documents(self, docs: list[dict[str, Any]]) -> int:
        self.documents.extend(docs)
        return len(self.documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, tenant_id: str = "default", top_k: int = 3) -> list[dict[str, Any]]:
        allowed_docs = [
            d for d in self.documents
            if d.get("tenant_id", "public") in ("public", "default", tenant_id)
        ]
        if not allowed_docs:
            return []

        q_tokens = self._tokenize(query)
        q_counter = Counter(q_tokens)

        bm25_scores = []
        for idx, doc in enumerate(allowed_docs):
            d_tokens = self._tokenize(doc.get("content", "") + " " + doc.get("title", ""))
            d_counter = Counter(d_tokens)
            overlap = sum(min(q_counter[w], d_counter[w]) * 2.0 for w in q_counter if w in d_counter)
            bm25_scores.append((idx, overlap / max(len(d_tokens), 10)))

        dense_scores = []
        for idx, doc in enumerate(allowed_docs):
            d_counter = Counter(self._tokenize(doc.get("content", "")))
            inter = set(q_counter.keys()) & set(d_counter.keys())
            num = sum(q_counter[x] * d_counter[x] for x in inter)
            den = math.sqrt(sum(v**2 for v in q_counter.values())) * math.sqrt(sum(v**2 for v in d_counter.values()))
            dense_scores.append((idx, num / den if den else 0.0))

        bm25_ranked = [idx for idx, _ in sorted(bm25_scores, key=lambda x: x[1], reverse=True)]
        dense_ranked = [idx for idx, _ in sorted(dense_scores, key=lambda x: x[1], reverse=True)]

        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)

        top_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            doc = allowed_docs[i]
            results.append({
                "doc_id": doc.get("id", f"doc-{i}"),
                "title": doc.get("title", "Enterprise Knowledge Snippet"),
                "content": doc.get("content", ""),
                "rrf_score": round(rrf_scores[i], 5),
            })
        return results
