"""MCP Server Configuration Loader

Loads MCP server definitions from environment variables and optional JSON config file.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from livekit.agents import mcp

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    url: str
    auth_token: str | None = None
    transport: Literal["sse", "streamable_http"] | None = None
    timeout: float = 10.0


def load_mcp_config() -> list[MCPServerConfig]:
    """Load MCP server configurations from env vars and optional JSON file.

    n8n is loaded from environment variables (foundational).
    Additional MCP servers can be configured in mcp_servers.json.

    Environment variables:
        N8N_MCP_URL: n8n MCP server URL
        N8N_MCP_TOKEN: Bearer token for n8n (optional)
        N8N_MCP_TIMEOUT: Request timeout in seconds (optional, default 10.0)

    JSON file (mcp_servers.json):
        {
            "servers": [
                {
                    "name": "server_name",
                    "url": "http://...",
                    "token": "optional_token",
                    "transport": "sse" | "streamable_http",
                    "timeout": 10.0
                }
            ]
        }

    Returns:
        List of MCPServerConfig objects
    """
    servers = []

    # 1. n8n MCP Server from env (foundational)
    n8n_url = os.environ.get("N8N_MCP_URL")
    if n8n_url:
        servers.append(MCPServerConfig(
            name="n8n",
            url=n8n_url,
            auth_token=os.environ.get("N8N_MCP_TOKEN"),
            transport="streamable_http",  # n8n uses /http suffix which needs explicit transport
            timeout=float(os.environ.get("N8N_MCP_TIMEOUT", "10.0")),
        ))
        logger.debug(f"Loaded MCP server config: n8n ({n8n_url})")
    else:
        logger.info("N8N_MCP_URL not set - n8n MCP tools will not be available")

    # 2. Additional MCP servers from JSON config (optional)
    config_path = Path("mcp_servers.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
                for server in data.get("servers", []):
                    name = server.get("name")
                    url = server.get("url")
                    if not name or not url:
                        logger.warning(f"Skipping MCP server with missing name or url: {server}")
                        continue

                    servers.append(MCPServerConfig(
                        name=name,
                        url=url,
                        auth_token=server.get("token"),
                        transport=server.get("transport"),
                        timeout=server.get("timeout", 10.0),
                    ))
                    logger.debug(f"Loaded MCP server config from JSON: {name} ({url})")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse mcp_servers.json: {e}")
        except Exception as e:
            logger.error(f"Failed to load mcp_servers.json: {e}")

    if not servers:
        logger.warning("No MCP servers configured - no MCP tools will be available")

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
    logger.info(f"initialize_mcp_servers: processing {len(configs)} configs")

    for config in configs:
        try:
            logger.info(f"Initializing MCP server: {config.name} at {config.url}")
            
            # Pre-check: Skip HTTPS URLs that might have SSL issues
            # This is a workaround for self-signed certificates that cause crashes
            if config.url.startswith("https://") and ".home" in config.url:
                logger.warning(
                    f"Skipping MCP server {config.name} - HTTPS URL with .home domain "
                    f"may have SSL certificate issues. To enable, fix SSL certificate or use HTTP."
                )
                continue
            
            headers = {}
            if config.auth_token:
                headers["Authorization"] = f"Bearer {config.auth_token}"

            logger.info(f"Creating MCPServerHTTP for {config.name}...")
            server = mcp.MCPServerHTTP(
                url=config.url,
                headers=headers if headers else None,
                timeout=config.timeout,
            )

            # Set transport type if specified
            # Newer LiveKit versions use transport_type param, older use private attr
            if config.transport == "streamable_http":
                server._use_streamable_http = True
            elif config.transport == "sse":
                server._use_streamable_http = False
            # If transport not specified, let LiveKit auto-detect from URL

            logger.info(f"Calling server.initialize() for {config.name}...")
            # Add timeout to prevent hanging - use config timeout + 5 seconds buffer
            init_timeout = config.timeout + 5.0
            
            # Wrap in a task with comprehensive error handling
            async def safe_initialize():
                try:
                    await server.initialize()
                    return True, None
                except BaseException as e:  # Catch ALL exceptions including SystemExit, KeyboardInterrupt
                    return False, e
            
            try:
                # Use create_task to isolate the initialization
                init_task = asyncio.create_task(safe_initialize())
                try:
                    success, error = await asyncio.wait_for(init_task, timeout=init_timeout)
                    if success:
                        logger.info(f"server.initialize() completed for {config.name}")
                        servers[config.name] = server
                        logger.info(f"Initialized MCP server: {config.name}")
                    else:
                        # Error occurred during initialization
                        init_error = error
                        raise init_error
                except asyncio.TimeoutError:
                    # Cancel the task
                    init_task.cancel()
                    try:
                        await init_task
                    except asyncio.CancelledError:
                        pass
                    raise
            except asyncio.TimeoutError as timeout_err:
                logger.error(f"MCP server {config.name} initialization timed out after {init_timeout}s")
                # Continue with other servers instead of failing completely
                continue
            except BaseException as init_error:  # Catch ALL exceptions
                # Check if it's an SSL verification error (may be wrapped in RuntimeError)
                error_str = str(init_error).lower()
                error_repr = repr(init_error).lower()
                error_type = type(init_error).__name__
                
                # Check exception chain for underlying SSL errors
                has_ssl_error = False
                current_exc = init_error
                depth = 0
                while current_exc is not None and depth < 10:  # Limit depth to prevent infinite loops
                    exc_str = str(current_exc).lower()
                    exc_type = type(current_exc).__name__
                    if "ssl" in exc_str or "certificate" in exc_str or "cert" in exc_str or "certificate_verify_failed" in exc_str:
                        has_ssl_error = True
                        break
                    # Also check for httpx.ConnectError with SSL issues
                    if exc_type == "ConnectError" and ("ssl" in exc_str or "certificate" in exc_str):
                        has_ssl_error = True
                        break
                    # Check for RuntimeError about cancel scope with SSL errors in chain
                    if exc_type == "RuntimeError" and "cancel scope" in exc_str:
                        # Check if there's an SSL error in the context
                        if hasattr(current_exc, "__context__") and current_exc.__context__:
                            ctx_str = str(current_exc.__context__).lower()
                            if "ssl" in ctx_str or "certificate" in ctx_str:
                                has_ssl_error = True
                                break
                    # Move to next exception in chain
                    current_exc = getattr(current_exc, "__cause__", None) or getattr(current_exc, "__context__", None)
                    depth += 1
                
                if has_ssl_error or "ssl" in error_str or "certificate" in error_str or "cert" in error_str:
                    logger.warning(
                        f"MCP server {config.name} SSL verification failed (likely self-signed cert). "
                        f"Error type: {error_type}, Error: {init_error}. Continuing without this MCP server."
                    )
                    # Skip this server but continue with others
                    continue
                else:
                    # Log other errors but continue with other servers
                    logger.error(f"Failed to initialize MCP server {config.name}: {init_error}", exc_info=True)
                    continue

        except BaseException as e:  # Catch ALL exceptions including SystemExit, KeyboardInterrupt
            # Catch any other unexpected errors during server creation
            logger.error(f"Unexpected error initializing MCP server {config.name}: {e}", exc_info=True)
            continue

    logger.info(f"initialize_mcp_servers: returning {len(servers)} servers")
    return servers
