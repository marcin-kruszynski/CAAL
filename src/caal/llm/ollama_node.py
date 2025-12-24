"""Simplified Ollama LLM Node with think parameter support.

This module provides a custom llm_node implementation that bypasses LiveKit's
LLM wrapper to enable direct Ollama calls with the `think` parameter.

Key Features:
- Direct Ollama API calls with think=False for lower latency (Qwen3)
- Tool discovery from @function_tool methods and MCP servers
- Tool execution routing (agent methods, n8n workflows, MCP tools)
- Streaming responses for best UX

Usage:
    class MyAgent(Agent):
        async def llm_node(self, chat_ctx, tools, model_settings):
            async for chunk in ollama_llm_node(
                self, chat_ctx, model="qwen3:8b", think=False
            ):
                yield chunk
"""

import inspect
import logging
from collections.abc import AsyncIterable
from typing import Any

import ollama

from ..utils.formatting import strip_markdown_for_tts
from ..integrations.n8n import execute_n8n_workflow

logger = logging.getLogger(__name__)


class OllamaLLMNode:
    """Encapsulates Ollama LLM configuration and tool handling."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        think: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ):
        self.model = model
        self.think = think
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._tools_cache: list[dict] | None = None

    def _get_ollama_options(self) -> dict:
        """Get Ollama options dict."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


async def ollama_llm_node(
    agent,
    chat_ctx,
    model: str = "qwen3:8b",
    think: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    num_ctx: int = 8192,
) -> AsyncIterable[str]:
    """Custom LLM node using Ollama directly with think parameter support.

    This function should be called from an Agent's llm_node method override.

    Args:
        agent: The Agent instance (self)
        chat_ctx: Chat context from LiveKit
        model: Ollama model name
        think: Enable thinking mode (False for lower latency with Qwen3)
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling limit
        num_ctx: Context window size

    Yields:
        String chunks for TTS output

    Example:
        class MyAgent(Agent):
            async def llm_node(self, chat_ctx, tools, model_settings):
                async for chunk in ollama_llm_node(self, chat_ctx, think=False):
                    yield chunk
    """
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_ctx": num_ctx,
    }

    try:
        # Build messages from chat context
        messages = _build_messages_from_context(chat_ctx)

        # Discover tools from agent and MCP servers
        ollama_tools = await _discover_tools(agent)

        # If tools available, check for tool calls first (non-streaming)
        if ollama_tools:
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=ollama_tools,
                think=think,
                stream=False,
                options=options,
            )

            if hasattr(response, "message"):
                # Check for tool calls
                if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                    tool_calls = response.message.tool_calls
                    logger.info(f"Ollama returned {len(tool_calls)} tool call(s)")

                    # Track tool usage for frontend indicator
                    tool_names = [tc.function.name for tc in tool_calls]
                    tool_params = [tc.function.arguments for tc in tool_calls]

                    # Publish tool status immediately (bullshit detector!)
                    if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                        import asyncio
                        asyncio.create_task(agent._on_tool_status(True, tool_names, tool_params))

                    # Execute tools and get results
                    messages = await _execute_tool_calls(
                        agent, messages, tool_calls, response.message
                    )

                    # Stream follow-up response with tool results
                    followup = ollama.chat(
                        model=model,
                        messages=messages,
                        think=think,
                        stream=True,
                        options=options,
                    )

                    for chunk in followup:
                        if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                            if chunk.message.content:
                                yield strip_markdown_for_tts(chunk.message.content)
                    return

                # No tool calls - return content directly
                elif hasattr(response.message, "content") and response.message.content:
                    # Publish no-tool status immediately
                    if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                        import asyncio
                        asyncio.create_task(agent._on_tool_status(False, [], []))
                    yield strip_markdown_for_tts(response.message.content)
                    return

        # No tools or no tool calls - stream directly
        # Publish no-tool status immediately
        if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
            import asyncio
            asyncio.create_task(agent._on_tool_status(False, [], []))

        response_stream = ollama.chat(
            model=model,
            messages=messages,
            tools=None,
            think=think,
            stream=True,
            options=options,
        )

        for chunk in response_stream:
            if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                if chunk.message.content:
                    yield strip_markdown_for_tts(chunk.message.content)

    except Exception as e:
        logger.error(f"Error in ollama_llm_node: {e}", exc_info=True)
        yield f"I encountered an error: {e}"


def _build_messages_from_context(chat_ctx) -> list[dict]:
    """Build Ollama messages from LiveKit chat context."""
    messages = []

    for item in chat_ctx.items:
        item_type = type(item).__name__

        if item_type == "ChatMessage":
            messages.append({
                "role": item.role,
                "content": item.text_content,
            })
        elif item_type == "FunctionCall":
            try:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.id,
                        "function": {
                            "name": item.name,
                            "arguments": getattr(item, "arguments", {}) or {},
                        },
                    }],
                })
            except AttributeError:
                pass
        elif item_type == "FunctionCallOutput":
            try:
                messages.append({
                    "role": "tool",
                    "content": str(item.content),
                    "tool_call_id": item.tool_call_id,
                })
            except AttributeError:
                pass

    return messages


async def _discover_tools(agent) -> list[dict] | None:
    """Discover tools from agent methods and MCP servers.

    Tools are cached on the agent instance after first discovery to avoid
    redundant MCP API calls on every user utterance.
    """
    # Return cached tools if available
    if hasattr(agent, "_ollama_tools_cache") and agent._ollama_tools_cache is not None:
        return agent._ollama_tools_cache

    ollama_tools = []

    # Get @function_tool decorated methods from agent
    if hasattr(agent, "_tools") and agent._tools:
        for tool in agent._tools:
            if hasattr(tool, "__func__"):
                func = tool.__func__
                name = func.__name__
                description = func.__doc__ or ""
                sig = inspect.signature(func)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == str:
                            param_type = "string"
                        elif param.annotation == int:
                            param_type = "integer"
                        elif param.annotation == float:
                            param_type = "number"
                        elif param.annotation == bool:
                            param_type = "boolean"
                    properties[param_name] = {"type": param_type}
                    if param.default == inspect.Parameter.empty and param_name != "self":
                        required.append(param_name)

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })

    # Get MCP tools from Home Assistant
    if hasattr(agent, "_mcp") and agent._mcp:
        mcp_tools = await _get_mcp_tools(agent._mcp)
        ollama_tools.extend(mcp_tools)

    # Add n8n workflow tools
    if hasattr(agent, "_n8n_workflow_tools") and agent._n8n_workflow_tools:
        ollama_tools.extend(agent._n8n_workflow_tools)

    # Cache tools on agent and return
    result = ollama_tools if ollama_tools else None
    agent._ollama_tools_cache = result

    return result


async def _get_mcp_tools(mcp_server) -> list[dict]:
    """Get tools from an MCP server in Ollama format."""
    tools = []

    if not mcp_server or not hasattr(mcp_server, "_client") or not mcp_server._client:
        return tools

    try:
        tools_result = await mcp_server._client.list_tools()
        if hasattr(tools_result, "tools"):
            for mcp_tool in tools_result.tools:
                # Convert MCP schema to Ollama format
                parameters = {"type": "object", "properties": {}, "required": []}
                if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                    schema = mcp_tool.inputSchema
                    if isinstance(schema, dict):
                        parameters = schema
                    elif hasattr(schema, "properties"):
                        parameters["properties"] = schema.properties or {}
                        parameters["required"] = getattr(schema, "required", []) or []

                tools.append({
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": getattr(mcp_tool, "description", "") or "",
                        "parameters": parameters,
                    },
                })

        if tools:
            tool_names = [t["function"]["name"] for t in tools]
            logger.info(f"Loaded {len(tools)} MCP tools: {', '.join(tool_names)}")

    except Exception as e:
        logger.warning(f"Error getting MCP tools: {e}")

    return tools


async def _execute_tool_calls(
    agent,
    messages: list[dict],
    tool_calls: list,
    response_message: Any,
) -> list[dict]:
    """Execute tool calls and append results to messages."""

    # Add assistant message with tool calls
    tool_call_message = {
        "role": "assistant",
        "content": getattr(response_message, "content", "") or "",
        "tool_calls": [],
    }

    for tool_call in tool_calls:
        tool_call_message["tool_calls"].append({
            "id": getattr(tool_call, "id", ""),
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments or {},
            },
        })

    messages.append(tool_call_message)

    # Execute each tool
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments or {}
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            tool_result = await _execute_single_tool(agent, tool_name, arguments)
            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": getattr(tool_call, "id", None),
            })
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            messages.append({
                "role": "tool",
                "content": error_msg,
                "tool_call_id": getattr(tool_call, "id", None),
            })

    return messages


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    """Execute a single tool call."""

    # Check if it's an agent method
    if hasattr(agent, tool_name) and callable(getattr(agent, tool_name)):
        logger.info(f"Calling agent tool: {tool_name}")
        result = await getattr(agent, tool_name)(**arguments)
        logger.info(f"Agent tool {tool_name} completed")
        return result

    # Check if it's an n8n workflow
    if (
        hasattr(agent, "_n8n_workflow_name_map")
        and tool_name in agent._n8n_workflow_name_map
        and hasattr(agent, "_n8n_base_url")
        and agent._n8n_base_url
    ):
        logger.info(f"Calling n8n workflow: {tool_name}")
        workflow_name = agent._n8n_workflow_name_map[tool_name]
        result = await execute_n8n_workflow(agent._n8n_base_url, workflow_name, arguments)
        logger.info(f"n8n workflow {tool_name} completed")
        return result

    # Check MCP servers
    if hasattr(agent, "_mcp") and agent._mcp:
        result = await _call_mcp_tool(agent._mcp, tool_name, arguments)
        if result is not None:
            return result

    raise ValueError(f"Tool {tool_name} not found")


async def _call_mcp_tool(mcp_server, tool_name: str, arguments: dict) -> Any | None:
    """Call a tool on an MCP server.

    Calls the tool directly without checking if it exists first - the MCP
    server will return an error if the tool doesn't exist.
    """
    if not mcp_server or not hasattr(mcp_server, "_client"):
        return None

    try:
        logger.info(f"Calling MCP tool: {tool_name}")
        result = await mcp_server._client.call_tool(tool_name, arguments)

        # Check for errors
        if result.isError:
            text_contents = []
            for content in result.content:
                if hasattr(content, "text") and content.text:
                    text_contents.append(content.text)
            error_msg = f"MCP tool {tool_name} error: {text_contents}"
            logger.error(error_msg)
            return error_msg

        # Extract text content
        text_contents = []
        for content in result.content:
            if hasattr(content, "text") and content.text:
                text_contents.append(content.text)

        return "\n".join(text_contents) if text_contents else "Tool executed successfully"

    except Exception as e:
        logger.warning(f"Error calling MCP tool {tool_name}: {e}")

    return None
