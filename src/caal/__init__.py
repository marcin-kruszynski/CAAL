"""
CAAL - Voice Assistant
======================

A modular voice assistant with n8n workflow integrations and local LLM support.

Core Components:
    LlamaCppLLM: llama.cpp server LLM with OpenAI-compatible API

STT/TTS:
    - Wyoming protocol STT (faster-whisper-wyoming)
    - Wyoming protocol TTS (piper-wyoming)

Integrations:
    n8n: Workflow discovery and execution via n8n MCP

Example:
    >>> from caal import LlamaCppLLM
    >>> from caal.integrations import load_mcp_config
    >>>
    >>> llm = LlamaCppLLM(model="gpt-oss-20b-mxfp4")
    >>> mcp_configs = load_mcp_config()

Repository: https://github.com/CoreWorxLab/caal
License: MIT
"""

__version__ = "0.1.0"
__author__ = "CoreWorxLab"

from .llm import LlamaCppLLM

__all__ = [
    "LlamaCppLLM",
    "__version__",
]
