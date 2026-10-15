"""17-Layer System Design Production Gateway Server (FastAPI + Google ADK Runner)."""

import json
import os
import time
import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app import active_cartridge, root_agent
from app.core.guardrails import EnterpriseGuardrailEngine
from app.core.idempotency import IdempotencyLedger
from app.core.msi_router import MinimumSufficientIntelligenceRouter
from app.core.observability import SystemMetricsStore
from app.core.semantic_cache import SemanticCacheEngine
from app.core.traffic_controller import EnterpriseTrafficController

app = FastAPI(
    title=f"{getattr(active_cartridge, 'AGENT_NAME', 'Universal ADK')} — 17-Layer System Design Gateway",
    description=getattr(active_cartridge, "AGENT_DESCRIPTION", ""),
    version="3.0.0",
)

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="universal_adk_v3", session_service=session_service)
traffic_controller = EnterpriseTrafficController(rate_per_minute=60, daily_token_cap=500_000)
semantic_cache = SemanticCacheEngine(similarity_threshold=0.88)
msi_router = MinimumSufficientIntelligenceRouter()
idempotency_ledger = IdempotencyLedger()
guardrail_engine = EnterpriseGuardrailEngine(active_cartridge)
metrics_store = SystemMetricsStore()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=6000, description="User message or task")
    session_id: str = Field(default="default-session", description="Multi-turn session ID")
    tenant_id: str = Field(default="default", description="Multi-tenant isolation ID")
    user_id: str = Field(default="enterprise-user", description="Authenticated user ID")
    bypass_cache: bool = Field(default=False, description="Force fresh LLM generation")


class CdcInvalidateRequest(BaseModel):
    tenant_id: str | None = None
    keyword: str | None = None


@app.middleware("http")
async def global_production_shield(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", f"trace-{uuid.uuid4().hex[:12]}")
    start = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Latency-Ms"] = str(round((time.perf_counter() - start) * 1000, 2))
        return response
    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        metrics_store.record_request(elapsed, 0, "TIER_2_STANDARD_FLASH", recovered_error=True)
        return JSONResponse(
            status_code=200,
            content={
                "status": "RECOVERED_BY_GLOBAL_SHIELD",
                "trace_id": trace_id,
                "message": "Request handled by fault-tolerant production shield.",
                "diagnostic": str(exc),
            },
        )


async def _execute_adk_turn(user_id: str, session_id: str, query_text: str) -> str:
    existing = await session_service.get_session(
        app_name="universal_adk_v3", user_id=user_id, session_id=session_id
    )
    if existing is None:
        await session_service.create_session(
            app_name="universal_adk_v3", user_id=user_id, session_id=session_id
        )

    content = types.Content(role="user", parts=[types.Part.from_text(text=query_text)])
    final_text_parts: list[str] = []

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text_parts.append(part.text)

    raw_answer = "\n".join(final_text_parts).strip() or "Task completed."
    return guardrail_engine.scrub_text(raw_answer)


@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    x_idempotency_key: str | None = Header(default=None),
):
    start = time.perf_counter()

    replay = idempotency_ledger.get(x_idempotency_key)
    if replay is not None:
        return replay

    est_tokens = max(len(req.query) // 3, 50)
    traffic = traffic_controller.check_and_consume(req.tenant_id, req.user_id, est_tokens)
    if not traffic["allowed"]:
        return JSONResponse(status_code=429, content={"status": "THROTTLED", "detail": traffic["reason"]})

    if not req.bypass_cache:
        cached = semantic_cache.lookup(req.tenant_id, req.query)
        if cached:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            metrics_store.record_request(elapsed, 0, "CACHE_HIT")
            payload = {
                "status": "SUCCESS",
                "agent": root_agent.name,
                "routing_tier": "SEMANTIC_CACHE_HIT",
                "similarity_score": cached["similarity_score"],
                "response": cached["cached_response"],
                "latency_ms": elapsed,
            }
            idempotency_ledger.save(x_idempotency_key, payload)
            return payload

    msi_rules = getattr(active_cartridge, "MSI_DETERMINISTIC_RULES", {})
    route_decision = msi_router.route(req.query, msi_rules)
    if route_decision["direct_response"] is not None:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        metrics_store.record_request(elapsed, 5, "TIER_1_DETERMINISTIC")
        payload = {
            "status": "SUCCESS",
            "agent": root_agent.name,
            "routing_tier": route_decision["tier"],
            "model_used": route_decision["model"],
            "response": route_decision["direct_response"],
            "latency_ms": elapsed,
        }
        semantic_cache.store(req.tenant_id, req.query, route_decision["direct_response"])
        idempotency_ledger.save(x_idempotency_key, payload)
        return payload

    answer = await _execute_adk_turn(req.user_id, req.session_id, req.query)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    total_tokens = est_tokens + max(len(answer) // 3, 50)
    metrics_store.record_request(elapsed, total_tokens, route_decision["tier"])

    semantic_cache.store(req.tenant_id, req.query, answer)
    payload = {
        "status": "SUCCESS",
        "agent": root_agent.name,
        "routing_tier": route_decision["tier"],
        "model_used": root_agent.model,
        "response": answer,
        "latency_ms": elapsed,
        "token_quota_status": traffic,
    }
    idempotency_ledger.save(x_idempotency_key, payload)
    return payload


@app.post("/chat/stream")
async def chat_stream_sse(req: ChatRequest):
    async def event_generator():
        result = await chat_endpoint(req, x_idempotency_key=None)
        text = result.get("response", "")
        words = text.split(" ")
        for i in range(0, len(words), 4):
            chunk = " ".join(words[i:i + 4]) + " "
            yield f"data: {json.dumps({'token_chunk': chunk, 'agent': root_agent.name})}\n\n"
        yield f"data: {json.dumps({'event': 'EOS', 'latency_ms': result.get('latency_ms')})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/cache/invalidate")
async def invalidate_semantic_cache(req: CdcInvalidateRequest):
    evicted = semantic_cache.invalidate_cdc(tenant_id=req.tenant_id, keyword=req.keyword)
    return {"status": "CDC_CACHE_INVALIDATED", "entries_evicted": evicted}


@app.post("/eval/run")
async def run_golden_dataset_evaluation():
    golden_cases = getattr(active_cartridge, "GOLDEN_EVAL_DATASET", [])
    results = []
    passed = 0
    for case in golden_cases:
        resp = await chat_endpoint(
            ChatRequest(query=case["query"], bypass_cache=True), x_idempotency_key=None
        )
        ans = resp.get("response", "").lower()
        expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]
        matched = [kw for kw in expected_kw if kw in ans]
        score = round(len(matched) / max(len(expected_kw), 1), 2)
        if score >= 0.5:
            passed += 1
        results.append({
            "query": case["query"],
            "score": score,
            "matched_keywords": matched,
            "routing_tier": resp.get("routing_tier"),
        })
    total = max(len(golden_cases), 1)
    return {
        "cartridge": root_agent.name,
        "total_cases": len(golden_cases),
        "pass_rate": round(passed / total, 2),
        "cases": results,
    }


@app.get("/metrics")
async def get_system_design_metrics():
    return {
        "active_cartridge": root_agent.name,
        "observability": metrics_store.summary(semantic_cache.hits, semantic_cache.misses),
    }


@app.get("/healthz")
async def healthz():
    return {
        "status": "HEALTHY",
        "active_agent": root_agent.name,
        "model": root_agent.model,
        "sub_agents_count": len(getattr(root_agent, "sub_agents", []) or []),
    }


@app.get("/readyz")
async def readyz():
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE")
    return {
        "status": "READY" if has_key else "MISSING_CREDENTIALS_WARNING",
        "api_key_configured": has_key,
        "rag_documents_loaded": len(getattr(active_cartridge, "KNOWLEDGE_BASE_DOCS", [])),
        "active_cartridge": root_agent.name,
    }
