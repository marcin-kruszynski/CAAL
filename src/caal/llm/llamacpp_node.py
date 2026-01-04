"""llama.cpp LLM Node with OpenAI-compatible API.

This module provides a custom llm_node implementation that uses llama.cpp server
via OpenAI-compatible API.

Key Features:
- Direct OpenAI-compatible API calls to llama.cpp server
- Tool discovery from @function_tool methods and MCP servers
- Tool execution routing (agent methods, n8n workflows, MCP tools)
- Streaming responses for best UX

Usage:
    class MyAgent(Agent):
        async def llm_node(self, chat_ctx, tools, model_settings):
            async for chunk in llamacpp_llm_node(
                self, chat_ctx, model="gpt-oss-20b-mxfp4"
            ):
                yield chunk
"""

import inspect
import json
import logging
import time
import asyncio
from collections.abc import AsyncIterable
from typing import Any

from openai import AsyncOpenAI

from ..utils.formatting import strip_markdown_for_tts
from ..integrations.n8n import execute_n8n_workflow
from ..ndjson_log import enabled as ndjson_enabled
from ..ndjson_log import log_event

logger = logging.getLogger(__name__)


class ToolDataCache:
    """Caches recent tool response data for context injection.

    Tool responses often contain structured data (IDs, arrays) that the LLM
    needs for follow-up calls. This cache preserves that data separately
    from chat history and injects it into context on each LLM call.
    """

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self._cache: list[dict] = []

    def add(self, tool_name: str, data: Any) -> None:
        """Add tool response data to cache."""
        entry = {"tool": tool_name, "data": data, "timestamp": time.time()}
        self._cache.append(entry)
        if len(self._cache) > self.max_entries:
            self._cache.pop(0)  # Remove oldest

    def get_context_message(self) -> str | None:
        """Format cached data as context string for LLM injection."""
        if not self._cache:
            return None
        parts = ["Recent tool response data for reference:"]
        for entry in self._cache:
            parts.append(f"\n{entry['tool']}: {json.dumps(entry['data'])}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class LlamaCppLLMNode:
    """Encapsulates llama.cpp LLM configuration and tool handling."""

    def __init__(
        self,
        model: str = "gpt-oss-20b-mxfp4",
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._tools_cache: list[dict] | None = None


async def llamacpp_llm_node(
    agent,
    chat_ctx,
    model: str = "gpt-oss-20b-mxfp4",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    num_ctx: int = 8192,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> AsyncIterable[str]:
    """Custom LLM node using llama.cpp server via OpenAI-compatible API.

    This function should be called from an Agent's llm_node method override.

    Args:
        agent: The Agent instance (self)
        chat_ctx: Chat context from LiveKit
        model: Model name
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling limit (may not be supported by all servers)
        num_ctx: Context window size (may not be supported by all servers)
        tool_data_cache: Cache for structured tool response data
        max_turns: Max conversation turns to keep in sliding window

    Yields:
        String chunks for TTS output

    Example:
        class MyAgent(Agent):
            async def llm_node(self, chat_ctx, tools, model_settings):
                async for chunk in llamacpp_llm_node(self, chat_ctx):
                    yield chunk
    """
    # Get base_url from agent's LLM instance
    base_url = getattr(agent.llm, "base_url", "http://llama.home/v1")

    # (debug instrumentation removed)

    # Initialize Async OpenAI client for llama.cpp server
    # IMPORTANT: this function is async; using the sync OpenAI client here can block the event loop
    # and prevent TTS/audio playout from running.
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",  # llama.cpp doesn't require auth
    )

    async def _create_with_timeout(params: dict, *, purpose: str, timeout_s: float = 20.0):
        """Call llama.cpp with explicit timeout."""
        t0 = time.time()
        log_event(
            hypothesis_id="L",
            location="llamacpp_node.py:_create_with_timeout",
            message="LLM request start",
            data={
                "purpose": purpose,
                "stream": bool(params.get("stream")),
                "has_tools": "tools" in params,
            },
        )
        try:
            resp = await asyncio.wait_for(client.chat.completions.create(**params), timeout=timeout_s)
            log_event(
                hypothesis_id="L",
                location="llamacpp_node.py:_create_with_timeout",
                message="LLM request ok",
                data={"purpose": purpose, "elapsed_ms": int((time.time() - t0) * 1000)},
            )
            return resp
        except asyncio.TimeoutError:
            log_event(
                hypothesis_id="L",
                location="llamacpp_node.py:_create_with_timeout",
                message="LLM request timeout",
                data={"purpose": purpose, "timeout_s": timeout_s},
            )
            raise

    try:
        _t0 = time.time()
        # Build messages from chat context with sliding window
        messages = _build_messages_from_context(
            chat_ctx,
            tool_data_cache=tool_data_cache,
            max_turns=max_turns,
        )

        # If chat_ctx contains consecutive user messages, the LLM may "continue" answering
        # the earlier one (especially after tool calls) because the assistant reply may not
        # be persisted into chat_ctx by the voice pipeline. We bridge this by inserting the
        # last assistant reply (tracked on the agent) between the two user messages.
        try:
            last_assistant = getattr(agent, "_caal_last_assistant_reply", None)
            if (
                isinstance(last_assistant, str)
                and last_assistant
                and len(messages) >= 2
                and messages[-1].get("role") == "user"
                and messages[-2].get("role") == "user"
            ):
                messages.insert(len(messages) - 1, {"role": "assistant", "content": last_assistant})
                log_event(
                    hypothesis_id="CTX",
                    location="llamacpp_node.py:llamacpp_llm_node",
                    message="Inserted cached assistant reply between consecutive user messages",
                    data={"assistant_preview": last_assistant[:200]},
                )
        except Exception:
            pass

        # Discover tools from agent and MCP servers
        openai_tools = await _discover_tools(agent)
        # (debug instrumentation removed)
        log_event(
            hypothesis_id="L",
            location="llamacpp_node.py:llamacpp_llm_node",
            message="LLM node entered",
            data={"model": model, "messages_len": len(messages), "tools_len": len(openai_tools)},
        )
        # Snapshot what the LLM sees at the tail of context (useful for debugging "wrong answer" issues).
        if ndjson_enabled():
            try:
                last_user = ""
                last_roles = [m.get("role") for m in messages[-8:]]
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user = (m.get("content") or "")[:200]
                        break
                log_event(
                    hypothesis_id="CTX",
                    location="llamacpp_node.py:llamacpp_llm_node",
                    message="Context snapshot",
                    data={
                        "last_roles": last_roles,
                        "last_user_preview": last_user,
                        "last_msg_role": (messages[-1].get("role") if messages else None),
                    },
                )
            except Exception:
                pass

        # Prepare API parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        # If tools available, check for tool calls first (non-streaming)
        if openai_tools:
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

            try:
                response = await _create_with_timeout(api_params, purpose="toolcheck", timeout_s=20.0)
            except asyncio.TimeoutError:
                # Fallback: retry without tools (prevents hanging forever on tool-enabled requests)
                api_params.pop("tools", None)
                api_params.pop("tool_choice", None)
                response = await _create_with_timeout(api_params, purpose="toolcheck_fallback_no_tools", timeout_s=20.0)

            # Check for tool calls
            choice = response.choices[0] if response.choices else None
            if choice and choice.message.tool_calls:
                tool_calls = choice.message.tool_calls
                logger.info(f"llama.cpp returned {len(tool_calls)} tool call(s)")

                # Track tool usage for frontend indicator
                tool_names = [tc.function.name for tc in tool_calls]
                tool_params = []
                for tc in tool_calls:
                    # Parse JSON arguments
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_params.append(args)

                # Publish tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    asyncio.create_task(agent._on_tool_status(True, tool_names, tool_params))

                # Execute tools and get results (cache structured data)
                messages = await _execute_tool_calls(
                    agent, messages, tool_calls, choice.message,
                    tool_data_cache=tool_data_cache,
                )

                # Follow-up response after tool results.
                #
                # IMPORTANT: Some OpenAI-compatible servers don't reliably stream delta.content.
                # If we stream and get no content, the assistant reply isn't persisted anywhere
                # (chat_ctx may not capture assistant messages), which can cause the next user turn
                # to look like consecutive user messages and the model "continues" the prior task.
                # Use non-stream followup to reliably capture full assistant reply.
                followup_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                }
                _t0_follow = time.time()
                follow_resp = await _create_with_timeout(followup_params, purpose="followup", timeout_s=20.0)
                content = ""
                raw_content = ""
                finish_reason = None
                has_tool_calls = False
                role = None
                try:
                    ch0 = follow_resp.choices[0] if getattr(follow_resp, "choices", None) else None
                    finish_reason = getattr(ch0, "finish_reason", None)
                    msg0 = getattr(ch0, "message", None)
                    role = getattr(msg0, "role", None)
                    has_tool_calls = bool(getattr(msg0, "tool_calls", None))
                    raw_content = getattr(msg0, "content", None) or ""
                    content = raw_content.strip()
                except Exception:
                    content = ""

                log_event(
                    hypothesis_id="TH",
                    location="llamacpp_node.py:llamacpp_llm_node",
                    message="followup non-stream finished",
                    data={
                        "elapsed_ms": int((time.time() - _t0_follow) * 1000),
                        "finish_reason": str(finish_reason)[:40] if finish_reason is not None else None,
                        "role": str(role)[:24] if role is not None else None,
                        "has_tool_calls": has_tool_calls,
                        "raw_content_len": len(raw_content),
                        "raw_content_preview": raw_content[:160],
                        "content_preview": content[:160],
                        "messages_len": len(messages),
                        "tail_roles": [m.get("role") for m in messages[-5:]],
                        "tail_tool_content_lens": [
                            len(m.get("content") or "") for m in messages[-5:] if m.get("role") == "tool"
                        ][:3],
                    },
                )
                try:
                    agent._caal_last_assistant_reply = content
                except Exception:
                    pass
                if content:
                    yield strip_markdown_for_tts(content)
                    return

                # Fallback: if the followup completion comes back empty (observed with some OpenAI-compatible servers),
                # speak the last tool result directly so the user still gets an answer.
                tool_fallback = ""
                try:
                    for m in reversed(messages):
                        if m.get("role") == "tool":
                            tool_fallback = (m.get("content") or "").strip()
                            break
                except Exception:
                    tool_fallback = ""

                log_event(
                    hypothesis_id="TH",
                    location="llamacpp_node.py:llamacpp_llm_node",
                    message="followup empty: using tool fallback",
                    data={
                        "tool_fallback_len": len(tool_fallback),
                        "tool_fallback_preview": tool_fallback[:160],
                    },
                )

                if tool_fallback:
                    # Keep it reasonably short for TTS.
                    yield strip_markdown_for_tts(tool_fallback[:800])
                else:
                    yield "Okay."
                return

            # No tool calls - return content directly
            elif choice and choice.message.content:
                # Persist assistant reply for next turn (used to bridge missing assistant messages in chat_ctx)
                try:
                    agent._caal_last_assistant_reply = choice.message.content or ""
                except Exception:
                    pass
                log_event(
                    hypothesis_id="L",
                    location="llamacpp_node.py:llamacpp_llm_node",
                    message="Toolcheck returned direct content",
                    data={"content_preview": (choice.message.content or "")[:160]},
                )
                # Publish no-tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    asyncio.create_task(agent._on_tool_status(False, [], []))
                yield strip_markdown_for_tts(choice.message.content)
                return

        # No tools or no tool calls - stream directly
        # Publish no-tool status immediately
        if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
            asyncio.create_task(agent._on_tool_status(False, [], []))

        stream_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        try:
            response_stream = await _create_with_timeout(stream_params, purpose="main_stream", timeout_s=20.0)
        except asyncio.TimeoutError:
            # Fallback: retry without tools if the tools-enabled request times out for some reason
            stream_params.pop("tools", None)
            stream_params.pop("tool_choice", None)
            response_stream = await _create_with_timeout(stream_params, purpose="main_stream_fallback_no_tools", timeout_s=20.0)

        _first = True
        _t_stream0 = time.time()
        _out_parts: list[str] = []
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if _first:
                    _first = False
                    log_event(
                        hypothesis_id="TH",
                        location="llamacpp_node.py:llamacpp_llm_node",
                        message="main stream first chunk",
                        data={},
                    )
                piece = strip_markdown_for_tts(chunk.choices[0].delta.content)
                _out_parts.append(piece)
                yield piece
        log_event(
            hypothesis_id="TH",
            location="llamacpp_node.py:llamacpp_llm_node",
            message="main stream finished",
            data={"elapsed_ms": int((time.time() - _t_stream0) * 1000)},
        )
        # Persist assistant reply for next turn (used to bridge missing assistant messages in chat_ctx)
        try:
            agent._caal_last_assistant_reply = "".join(_out_parts)
        except Exception:
            pass
        # (debug instrumentation removed)

    except Exception as e:
        logger.error(f"Error in llamacpp_llm_node: {e}", exc_info=True)
        yield f"I encountered an error: {e}"


def _build_messages_from_context(
    chat_ctx,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> list[dict]:
    """Build OpenAI-compatible messages with sliding window and tool data context.

    Message order:
    1. System prompt (always first, never trimmed)
    2. Tool data context (injected from cache)
    3. Chat history (sliding window applied)

    Args:
        chat_ctx: LiveKit chat context
        tool_data_cache: Cache of recent tool response data
        max_turns: Max conversation turns to keep (1 turn = user + assistant)
    """
    system_prompt = None
    chat_messages = []

    for item in chat_ctx.items:
        item_type = type(item).__name__

        if item_type == "ChatMessage":
            msg = {"role": item.role, "content": item.text_content}
            if item.role == "system":
                system_prompt = msg
            else:
                chat_messages.append(msg)
        elif item_type == "FunctionCall":
            try:
                chat_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": json.dumps(getattr(item, "arguments", {}) or {}),
                        },
                    }],
                })
            except AttributeError:
                pass
        elif item_type == "FunctionCallOutput":
            try:
                chat_messages.append({
                    "role": "tool",
                    "content": str(item.content),
                    "tool_call_id": item.tool_call_id,
                })
            except AttributeError:
                pass

    # Build final message list
    messages = []

    # 1. System prompt always first
    if system_prompt:
        messages.append(system_prompt)

    # 2. Inject tool data context
    if tool_data_cache:
        context = tool_data_cache.get_context_message()
        if context:
            messages.append({"role": "system", "content": context})

    # 3. Apply sliding window to chat history
    # max_turns * 2 accounts for user + assistant pairs
    max_messages = max_turns * 2
    if len(chat_messages) > max_messages:
        trimmed = len(chat_messages) - max_messages
        chat_messages = chat_messages[-max_messages:]
        logger.debug(f"Sliding window: trimmed {trimmed} old messages")

    messages.extend(chat_messages)
    return messages


async def _discover_tools(agent) -> list[dict] | None:
    """Discover tools from agent methods and MCP servers.

    Tools are cached on the agent instance after first discovery to avoid
    redundant MCP API calls on every user utterance.
    """
    # Return cached tools if available
    if hasattr(agent, "_llamacpp_tools_cache") and agent._llamacpp_tools_cache is not None:
        return agent._llamacpp_tools_cache

    openai_tools = []

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

                openai_tools.append({
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

    # Get MCP tools from all configured servers (except n8n)
    # n8n uses webhook-based workflow discovery, not direct MCP tools
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        for server_name, server in agent._caal_mcp_servers.items():
            # Skip n8n - it uses workflow discovery via _n8n_workflow_tools
            if server_name == "n8n":
                continue

            mcp_tools = await _get_mcp_tools(server)
            # Prefix tools with server name to avoid collisions
            for tool in mcp_tools:
                original_name = tool["function"]["name"]
                tool["function"]["name"] = f"{server_name}__{original_name}"
            openai_tools.extend(mcp_tools)
            if mcp_tools:
                logger.info(f"Added {len(mcp_tools)} tools from MCP server: {server_name}")

    # Add n8n workflow tools (webhook-based execution, separate from MCP)
    if hasattr(agent, "_n8n_workflow_tools") and agent._n8n_workflow_tools:
        openai_tools.extend(agent._n8n_workflow_tools)

    # Cache tools on agent and return
    result = openai_tools if openai_tools else None
    agent._llamacpp_tools_cache = result

    return result


async def _get_mcp_tools(mcp_server) -> list[dict]:
    """Get tools from an MCP server in OpenAI format."""
    tools = []

    if not mcp_server or not hasattr(mcp_server, "_client") or not mcp_server._client:
        return tools

    try:
        tools_result = await mcp_server._client.list_tools()
        if hasattr(tools_result, "tools"):
            for mcp_tool in tools_result.tools:
                # Convert MCP schema to OpenAI format
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
    tool_data_cache: ToolDataCache | None = None,
) -> list[dict]:
    """Execute tool calls and append results to messages.

    Args:
        agent: The agent instance
        messages: Current message list to append to
        tool_calls: List of tool calls from LLM response
        response_message: The original LLM response message
        tool_data_cache: Optional cache to store structured tool response data
    """

    # Add assistant message with tool calls
    tool_call_message = {
        "role": "assistant",
        "content": getattr(response_message, "content", "") or "",
        "tool_calls": [],
    }

    for tool_call in tool_calls:
        tool_call_message["tool_calls"].append({
            "id": getattr(tool_call, "id", ""),
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments if isinstance(tool_call.function.arguments, str) else json.dumps(tool_call.function.arguments),
            },
        })

    messages.append(tool_call_message)

    # Execute each tool
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        # Parse arguments (OpenAI sends as JSON string)
        try:
            arguments = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else (tool_call.function.arguments or {})
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
        _t0 = time.time()
        log_event(
            hypothesis_id="TH",
            location="llamacpp_node.py:_execute_tool_calls",
            message="tool start",
            data={"tool": tool_name, "args_keys": list(arguments.keys())[:12]},
        )

        try:
            tool_result = await _execute_single_tool(agent, tool_name, arguments)
            log_event(
                hypothesis_id="TH",
                location="llamacpp_node.py:_execute_tool_calls",
                message="tool ok",
                data={"tool": tool_name, "elapsed_ms": int((time.time() - _t0) * 1000)},
            )

            # Cache structured data if present
            if tool_data_cache and isinstance(tool_result, dict):
                # Look for common data fields, otherwise cache the whole result
                data = tool_result.get("data") or tool_result.get("results") or tool_result
                tool_data_cache.add(tool_name, data)
                logger.debug(f"Cached tool data for {tool_name}")

            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": getattr(tool_call, "id", None),
            })
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            log_event(
                hypothesis_id="TH",
                location="llamacpp_node.py:_execute_tool_calls",
                message="tool exception",
                data={"tool": tool_name, "elapsed_ms": int((time.time() - _t0) * 1000), "error": str(e)[:200]},
            )
            messages.append({
                "role": "tool",
                "content": error_msg,
                "tool_call_id": getattr(tool_call, "id", None),
            })

    return messages


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    """Execute a single tool call.

    Routing priority:
    1. Agent methods (@function_tool decorated)
    2. n8n workflows (webhook-based execution)
    3. MCP servers (with server_name__tool_name prefix parsing)
    """

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

    # Check MCP servers (with multi-server routing)
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        # Parse server name from prefixed tool name
        # Format: server_name__actual_tool (double underscore separator)
        if "__" in tool_name:
            server_name, actual_tool = tool_name.split("__", 1)
        else:
            # Unprefixed tools default to n8n server
            server_name, actual_tool = "n8n", tool_name

        if server_name in agent._caal_mcp_servers:
            server = agent._caal_mcp_servers[server_name]
            result = await _call_mcp_tool(server, actual_tool, arguments)
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

