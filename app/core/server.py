"""17-Layer System Design Production Gateway Server + Built-In ADK Agent Studio UI."""

import json
import os
import time
import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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


@app.get("/agent/info")
async def get_agent_info():
    """Returns dynamic cartridge metadata so the UI automatically adapts to any 1-file swap."""
    custom_tools = [getattr(t, "__name__", str(t)) for t in getattr(active_cartridge, "CUSTOM_TOOLS", [])]
    base_tools = ["universal_calculator", "get_current_time", "search_enterprise_rag_knowledge"]
    sub_agents = [
        {"name": s.get("name"), "description": s.get("description", "")}
        for s in getattr(active_cartridge, "SUB_AGENTS", [])
    ]
    eval_queries = [c.get("query") for c in getattr(active_cartridge, "GOLDEN_EVAL_DATASET", [])]
    return {
        "agent_name": root_agent.name,
        "description": getattr(active_cartridge, "AGENT_DESCRIPTION", ""),
        "model": root_agent.model,
        "tools": base_tools + custom_tools,
        "sub_agents": sub_agents,
        "rag_docs_count": len(getattr(active_cartridge, "KNOWLEDGE_BASE_DOCS", [])),
        "sample_prompts": eval_queries,
    }


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


AGENT_STUDIO_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Universal ADK Studio — 17-Layer System Design Console</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', -apple-system, sans-serif; }
    .glass { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
    .prose pre { background: #0f172a; padding: 10px; border-radius: 8px; overflow-x: auto; }
    .prose ul { list-style-type: disc; padding-left: 1.25rem; margin-top: 0.4rem; margin-bottom: 0.4rem; }
    .prose ol { list-style-type: decimal; padding-left: 1.25rem; margin-top: 0.4rem; margin-bottom: 0.4rem; }
    .prose p { margin-bottom: 0.5rem; }
    .prose strong { color: #38bdf8; }
  </style>
</head>
<body class="h-screen flex flex-col overflow-hidden">
  <header class="glass px-6 py-3 flex items-center justify-between border-b border-slate-800">
    <div class="flex items-center gap-3">
      <div class="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
      <h1 id="header-agent-name" class="text-lg font-bold tracking-wide text-white">Loading Agent Cartridge...</h1>
      <span id="header-model-badge" class="text-xs px-2.5 py-1 rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/30 font-mono">gemini</span>
    </div>
    <div class="flex items-center gap-3 text-xs">
      <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Swagger /docs ↗</a>
      <button onclick="runGoldenEval()" class="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition">🧪 Run Golden Eval</button>
      <button onclick="invalidateCache()" class="px-3 py-1.5 rounded-lg bg-amber-600/80 hover:bg-amber-500 text-white font-medium transition">🧹 CDC Evict Cache</button>
    </div>
  </header>

  <div class="flex-1 grid grid-cols-12 overflow-hidden">
    <aside class="col-span-3 glass border-r border-slate-800 p-4 overflow-y-auto flex flex-col gap-5">
      <div>
        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Active Type-C Cartridge</h2>
        <p id="agent-desc" class="text-xs text-slate-300 leading-relaxed"></p>
      </div>

      <div>
        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Multi-Agent Swarm Topology</h2>
        <div id="subagents-list" class="flex flex-col gap-2"></div>
      </div>

      <div>
        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Registered Resilient Tools</h2>
        <div id="tools-list" class="flex flex-wrap gap-1.5"></div>
      </div>

      <div class="border-t border-slate-800 pt-4 flex flex-col gap-2.5">
        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400">Gateway Context & Headers</h2>
        <div>
          <label class="text-[11px] text-slate-400">Tenant ID (ACL Isolation)</label>
          <input id="tenant-id" value="default" class="w-full mt-0.5 px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-xs text-white" />
        </div>
        <div>
          <label class="text-[11px] text-slate-400">Session ID (Multi-Turn Memory)</label>
          <input id="session-id" value="studio-session-01" class="w-full mt-0.5 px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-xs text-white" />
        </div>
        <div>
          <label class="text-[11px] text-slate-400">X-Idempotency-Key (Optional)</label>
          <input id="idem-key" placeholder="Auto-generated if empty" class="w-full mt-0.5 px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-xs text-white" />
        </div>
        <label class="flex items-center gap-2 text-xs text-slate-300 mt-1 cursor-pointer">
          <input type="checkbox" id="bypass-cache" class="rounded bg-slate-900 border-slate-700" />
          Bypass Semantic Cache
        </label>
      </div>
    </aside>

    <main class="col-span-6 flex flex-col bg-slate-950/60 overflow-hidden">
      <div id="sample-prompts" class="px-4 py-2.5 border-b border-slate-800/80 flex items-center gap-2 overflow-x-auto text-xs"></div>

      <div id="chat-box" class="flex-1 p-5 overflow-y-auto space-y-4">
        <div class="glass p-4 rounded-xl border border-slate-800 text-sm text-slate-300">
          👋 <strong>Welcome to the Universal ADK Production Studio!</strong><br/>
          Ask any question below or click a preset prompt above. Every message passes through the <strong>17-Layer Enterprise System Design Engine</strong> (Rate Limiter, Idempotency Shield, Semantic Cache, MSI Router, Hybrid RAG, and 360° DLP Guardrails).
        </div>
      </div>

      <form onsubmit="sendMessage(event)" class="p-4 glass border-t border-slate-800 flex gap-2">
        <input
          id="chat-input"
          type="text"
          placeholder="Ask your active agent swarm anything..."
          class="flex-1 px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-sm text-white focus:outline-none focus:border-sky-500"
          required
        />
        <button
          id="send-btn"
          type="submit"
          class="px-5 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-semibold text-sm transition"
        >
          Send →
        </button>
      </form>
    </main>

    <aside class="col-span-3 glass border-l border-slate-800 p-4 overflow-y-auto flex flex-col gap-4">
      <div class="flex items-center justify-between">
        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400">17-Layer Live Telemetry</h2>
        <button onclick="refreshMetrics()" class="text-xs text-sky-400 hover:underline">↻ Refresh</button>
      </div>

      <div class="grid grid-cols-2 gap-2.5">
        <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
          <div class="text-[11px] text-slate-400">P50 / P95 Latency</div>
          <div id="m-latency" class="text-base font-bold text-white mt-0.5">0 / 0 ms</div>
        </div>
        <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
          <div class="text-[11px] text-slate-400">Cache Hit Ratio</div>
          <div id="m-cache" class="text-base font-bold text-emerald-400 mt-0.5">0%</div>
        </div>
        <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
          <div class="text-[11px] text-slate-400">Total Requests</div>
          <div id="m-reqs" class="text-base font-bold text-sky-400 mt-0.5">0</div>
        </div>
        <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
          <div class="text-[11px] text-slate-400">Est. Token Cost</div>
          <div id="m-cost" class="text-base font-bold text-amber-400 mt-0.5">$0.0000</div>
        </div>
      </div>

      <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
        <div class="text-xs font-semibold text-slate-300 mb-2">MSI Routing Tier Distribution</div>
        <div id="m-tiers" class="space-y-1.5 text-xs text-slate-400"></div>
      </div>

      <div id="eval-output" class="hidden p-3 rounded-xl bg-indigo-950/40 border border-indigo-500/30 text-xs space-y-2"></div>
    </aside>
  </div>

  <script>
    async function loadAgentInfo() {
      const res = await fetch('/agent/info');
      const data = await res.json();
      document.getElementById('header-agent-name').textContent = data.agent_name;
      document.getElementById('header-model-badge').textContent = data.model;
      document.getElementById('agent-desc').textContent = data.description;

      const subList = document.getElementById('subagents-list');
      subList.innerHTML = '';
      if (data.sub_agents.length === 0) {
        subList.innerHTML = '<div class="text-xs text-slate-500">Single Root Agent Mode</div>';
      } else {
        data.sub_agents.forEach(sa => {
          subList.innerHTML += `
            <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div class="text-xs font-semibold text-sky-400">⚡ ${sa.name}</div>
              <div class="text-[11px] text-slate-400 mt-0.5">${sa.description}</div>
            </div>`;
        });
      }

      const toolsList = document.getElementById('tools-list');
      toolsList.innerHTML = '';
      data.tools.forEach(t => {
        toolsList.innerHTML += `<span class="px-2 py-1 rounded bg-slate-900 border border-slate-700 text-[11px] font-mono text-emerald-300">🔧 ${t}</span>`;
      });

      const promptsBar = document.getElementById('sample-prompts');
      promptsBar.innerHTML = '<span class="text-slate-500 font-semibold">Try:</span>';
      data.sample_prompts.forEach(q => {
        const btn = document.createElement('button');
        btn.className = 'px-2.5 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 whitespace-nowrap transition';
        btn.textContent = q;
        btn.onclick = () => {
          document.getElementById('chat-input').value = q;
          document.getElementById('send-btn').click();
        };
        promptsBar.appendChild(btn);
      });
    }

    async function refreshMetrics() {
      const res = await fetch('/metrics');
      const data = await res.json();
      const obs = data.observability;
      document.getElementById('m-latency').textContent = `${obs.p50_latency_ms} / ${obs.p95_latency_ms} ms`;
      document.getElementById('m-cache').textContent = `${Math.round(obs.semantic_cache_hit_ratio * 100)}%`;
      document.getElementById('m-reqs').textContent = obs.total_requests;
      document.getElementById('m-cost').textContent = `$${obs.estimated_cost_usd}`;

      const tiersDiv = document.getElementById('m-tiers');
      tiersDiv.innerHTML = '';
      for (const [k, v] of Object.entries(obs.routing_distribution)) {
        tiersDiv.innerHTML += `<div class="flex justify-between"><span>${k}</span><span class="font-mono text-white">${v}</span></div>`;
      }
    }

    function appendBubble(role, text, meta = null) {
      const box = document.getElementById('chat-box');
      const wrapper = document.createElement('div');
      wrapper.className = role === 'user' ? 'flex justify-end' : 'flex justify-start';

      const card = document.createElement('div');
      card.className = role === 'user'
        ? 'max-w-[80%] px-4 py-2.5 rounded-2xl bg-sky-600 text-white text-sm'
        : 'max-w-[90%] p-4 rounded-2xl glass border border-slate-800 text-sm text-slate-200 space-y-2';

      if (role === 'user') {
        card.textContent = text;
      } else {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'prose prose-invert max-w-none text-sm leading-relaxed';
        contentDiv.innerHTML = marked.parse(text || '');
        card.appendChild(contentDiv);

        if (meta) {
          const badgeBar = document.createElement('div');
          badgeBar.className = 'pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-[11px] font-mono text-slate-400';
          badgeBar.innerHTML = `
            <span class="px-2 py-0.5 rounded bg-slate-900 text-sky-300 border border-slate-700">Tier: ${meta.routing_tier || 'ADK'}</span>
            <span class="px-2 py-0.5 rounded bg-slate-900 text-emerald-300 border border-slate-700">⚡ ${meta.latency_ms} ms</span>
            ${meta._idempotent_replay ? '<span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">🔁 Idempotent Replay</span>' : ''}
          `;
          card.appendChild(badgeBar);
        }
      }

      wrapper.appendChild(card);
      box.appendChild(wrapper);
      box.scrollTop = box.scrollHeight;
    }

    async function sendMessage(e) {
      e.preventDefault();
      const input = document.getElementById('chat-input');
      const query = input.value.trim();
      if (!query) return;
      input.value = '';

      appendBubble('user', query);
      const btn = document.getElementById('send-btn');
      btn.disabled = true;
      btn.textContent = 'Thinking...';

      const headers = { 'Content-Type': 'application/json' };
      const idemKey = document.getElementById('idem-key').value.trim();
      if (idemKey) headers['X-Idempotency-Key'] = idemKey;

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers,
          body: JSON.stringify({
            query,
            tenant_id: document.getElementById('tenant-id').value || 'default',
            session_id: document.getElementById('session-id').value || 'studio-session-01',
            user_id: 'studio-user',
            bypass_cache: document.getElementById('bypass-cache').checked,
          }),
        });
        const data = await res.json();
        appendBubble('agent', data.response || data.detail || JSON.stringify(data), data);
        await refreshMetrics();
      } catch (err) {
        appendBubble('agent', 'Error communicating with gateway: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Send →';
      }
    }

    async function runGoldenEval() {
      const box = document.getElementById('eval-output');
      box.classList.remove('hidden');
      box.innerHTML = '⏳ Running Golden Benchmark Dataset...';
      const res = await fetch('/eval/run', { method: 'POST' });
      const data = await res.json();
      box.innerHTML = `
        <div class="font-bold text-indigo-300">✅ Golden Eval Pass Rate: ${Math.round(data.pass_rate * 100)}% (${data.total_cases} cases)</div>
        ${data.cases.map(c => `<div class="text-[11px] text-slate-300">• "${c.query}" → Score: <b>${c.score}</b> (${c.routing_tier})</div>`).join('')}
      `;
      await refreshMetrics();
    }

    async function invalidateCache() {
      const res = await fetch('/cache/invalidate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      alert(`CDC Cache Invalidation complete! Evicted ${data.entries_evicted} entries.`);
      await refreshMetrics();
    }

    loadAgentInfo();
    refreshMetrics();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_agent_studio_ui():
    """Serves the interactive ADK Agent Studio & 17-Layer System Design Console."""
    return HTMLResponse(content=AGENT_STUDIO_HTML)
