"""
Reasoning Integration for Agents
================================
Re-exports from AitherNode/reason_client.py for use by agents.

The reasoning client automatically checks if AitherReasoning (port 8093) is running
and gracefully degrades if it's not available.

Usage:
    from aither_adk.ai.reasoning import with_reasoning, get_reasoning_client, ThoughtType
    
    # In your agent's process loop:
    session = await with_reasoning("Aither", user_query)
    if session:  # Only logs if AitherReasoning is running
        await session.think("Analyzing the request...")
        # ... do work ...
        await session.conclude(response)
"""

import sys
import os

# Add AitherNode/lib to path for imports
_aithernode_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../AitherNode/lib'))
if _aithernode_lib_path not in sys.path:
    sys.path.insert(0, _aithernode_lib_path)

# Re-export everything from the canonical source
from reason_client import (
    # Core client
    ReasoningClient,
    get_reasoning_client,
    
    # Session management
    ReasoningSession,
    with_reasoning,
    reasoning_context,
    
    # Types
    ThoughtType,
    TokenInfo,
    LLMCall,
    
    # Interceptors for LLM calls
    LLMInterceptor,
    create_interceptor,
    
    # Decorators and utilities
    traced,
    stream_with_reasoning,
    capture_reasoning,
    
    # Constants
    REASON_URL,
)

__all__ = [
    'ReasoningClient',
    'get_reasoning_client',
    'ReasoningSession', 
    'with_reasoning',
    'reasoning_context',
    'ThoughtType',
    'TokenInfo',
    'LLMCall',
    'LLMInterceptor',
    'create_interceptor',
    'traced',
    'stream_with_reasoning',
    'capture_reasoning',
    'REASON_URL',
]
