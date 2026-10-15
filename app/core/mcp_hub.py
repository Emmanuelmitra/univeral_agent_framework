"""Fault-Tolerant Universal MCP Connector Hub with Circuit Breaker."""

import logging
import shutil
from typing import Any

logger = logging.getLogger("universal_agent.mcp_hub")


def load_resilient_mcp_tools(mcp_configs: list[dict[str, Any]]) -> list[Any]:
    loaded_toolsets = []
    if not mcp_configs:
        return loaded_toolsets

    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, SseServerParams
    except ImportError:
        logger.warning("google-adk MCPToolset not available; skipping external MCP servers safely.")
        return loaded_toolsets

    for server in mcp_configs:
        if not server.get("enabled", False):
            continue

        name = server.get("name", "unnamed_mcp")
        transport = server.get("transport", "stdio")

        try:
            if transport == "stdio":
                cmd = server.get("command", "uvx")
                if shutil.which(cmd) is None:
                    logger.warning("MCP Circuit Breaker: '%s' binary not found; skipping '%s'.", cmd, name)
                    continue
                params = StdioServerParameters(
                    command=cmd,
                    args=server.get("args", []),
                    env=server.get("env"),
                )
                loaded_toolsets.append(MCPToolset(connection_params=params))
            elif transport == "sse":
                params = SseServerParams(url=server["url"])
                loaded_toolsets.append(MCPToolset(connection_params=params))
        except Exception as exc:
            logger.error("MCP Circuit Breaker tripped for '%s': %s. Continuing safely.", name, exc)

    return loaded_toolsets
