import ast, datetime, operator, os
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from app import active_cartridge
from app.core.guardrails import EnterpriseGuardrailEngine
from app.core.hybrid_rag import HybridRagEngine
from app.core.mcp_hub import load_resilient_mcp_tools
from app.core.resilience import resilient_tool

rag_engine = HybridRagEngine(getattr(active_cartridge, "KNOWLEDGE_BASE_DOCS", []))

@resilient_tool
def universal_calculator(expression: str) -> dict:
    """Safely evaluates mathematical expressions (e.g. '1450 * 1.18', '2 ** 10')."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp): return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    parsed = ast.parse(expression.strip(), mode="eval")
    return {"expression": expression, "value": _eval(parsed.body)}

@resilient_tool
def get_current_time(timezone_name: str = "Asia/Kolkata") -> dict:
    """Returns current ISO timestamp for a given IANA timezone (e.g. 'UTC', 'Asia/Kolkata')."""
    try: tz = ZoneInfo(timezone_name)
    except Exception: tz = datetime.timezone.utc; timezone_name = "UTC (fallback)"
    return {"timezone": timezone_name, "timestamp": datetime.datetime.now(tz).isoformat()}

@resilient_tool
def search_enterprise_rag_knowledge(query: str, tenant_id: str = "default") -> dict:
    """Searches the Hybrid RAG Knowledge Base (BM25 + Dense RRF) for verified policies and runbooks."""
    chunks = rag_engine.search(query=query, tenant_id=tenant_id, top_k=3)
    return {"query": query, "tenant_id": tenant_id, "retrieved_chunks": chunks}

def build_agent_from_cartridge(cartridge=active_cartridge) -> Agent:
    guardrail_engine = EnterpriseGuardrailEngine(cartridge)
    raw_tools = getattr(cartridge, "CUSTOM_TOOLS", [])
    safe_custom_tools = [resilient_tool(t) for t in raw_tools]
    mcp_configs = getattr(cartridge, "MCP_SERVERS", [])
    mcp_toolsets = load_resilient_mcp_tools(mcp_configs)
    base_tools = [universal_calculator, get_current_time, search_enterprise_rag_knowledge]

    sub_agent_specs = getattr(cartridge, "SUB_AGENTS", [])
    built_sub_agents = []
    for spec in sub_agent_specs:
        sub_tools = [resilient_tool(t) for t in spec.get("tools", [])]
        built_sub_agents.append(Agent(
            name=spec["name"],
            model=spec.get("model", getattr(cartridge, "MODEL_NAME", "gemini-2.5-flash")),
            description=spec["description"],
            instruction=spec["instruction"],
            tools=base_tools + sub_tools,
            before_agent_callback=guardrail_engine.input_guardrail,
            before_tool_callback=guardrail_engine.before_tool_guardrail,
            after_tool_callback=guardrail_engine.after_tool_guardrail,
        ))

    system_prompt = os.getenv("OVERRIDE_SYSTEM_PROMPT", getattr(cartridge, "SYSTEM_PROMPT", "You are a helpful Enterprise AI Agent."))
    return Agent(
        name=getattr(cartridge, "AGENT_NAME", "universal_enterprise_agent"),
        model=os.getenv("ADK_MODEL_NAME", getattr(cartridge, "MODEL_NAME", "gemini-2.5-flash")),
        description=getattr(cartridge, "AGENT_DESCRIPTION", "Enterprise AI Agent."),
        instruction=system_prompt,
        tools=base_tools + safe_custom_tools + mcp_toolsets,
        sub_agents=built_sub_agents,
        before_agent_callback=guardrail_engine.input_guardrail,
        before_tool_callback=guardrail_engine.before_tool_guardrail,
        after_tool_callback=guardrail_engine.after_tool_guardrail,
    )
