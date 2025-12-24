"""MCP Server Configuration Loader

Loads MCP server definitions from environment variables.
"""

import os
import logging
from dataclasses import dataclass

from livekit.agents import mcp

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    url: str
    auth_token: str | None = None
    timeout: float = 10.0


def load_mcp_config() -> list[MCPServerConfig]:
    """Load MCP server configurations from environment variables.

    Environment variables:
        N8N_MCP_URL: n8n MCP server URL (required)
        N8N_MCP_TOKEN: Bearer token for n8n (optional)
        N8N_MCP_TIMEOUT: Request timeout in seconds (optional, default 10.0)

    Returns:
        List of MCPServerConfig objects
    """
    servers = []

    # n8n MCP Server
    n8n_url = os.environ.get("N8N_MCP_URL")
    if n8n_url:
        servers.append(MCPServerConfig(
            name="n8n-workflows",
            url=n8n_url,
            auth_token=os.environ.get("N8N_MCP_TOKEN"),
            timeout=float(os.environ.get("N8N_MCP_TIMEOUT", "10.0")),
        ))
        logger.debug(f"Loaded MCP server config: n8n-workflows ({n8n_url})")
    else:
        logger.warning("N8N_MCP_URL not set - no MCP tools will be available")

    return servers


async def initialize_mcp_servers(
    configs: list[MCPServerConfig]
) -> dict[str, mcp.MCPServerHTTP]:
    """Initialize MCP servers from config list.

    Args:
        configs: List of MCPServerConfig objects

    Returns:
        Dict mapping server name to initialized MCPServerHTTP instance
    """
    servers = {}

    for config in configs:
        try:
            headers = {}
            if config.auth_token:
                headers["Authorization"] = f"Bearer {config.auth_token}"

            server = mcp.MCPServerHTTP(
                url=config.url,
                headers=headers if headers else None,
                timeout=config.timeout,
            )

            # Force streamable HTTP transport for URLs ending in /http (like n8n)
            # LiveKit's MCPServerHTTP only auto-detects streamable HTTP for URLs ending in /mcp
            if config.url.endswith("/http"):
                server._use_streamable_http = True
                logger.debug(f"Forcing streamable HTTP transport for {config.name}")

            await server.initialize()
            servers[config.name] = server
            logger.debug(f"Initialized MCP server: {config.name}")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server {config.name}: {e}", exc_info=True)

    return servers
