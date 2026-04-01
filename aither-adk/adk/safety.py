"""Safety gates — prompt injection defense and output validation.

Lightweight port of AitherOS safety stack:
  - IntakeGuard: Fast regex scan for common injection patterns
  - OutputValidator: Check agent responses for safety issues
  - ContentFilter: Block/warn on dangerous content

All gates are NON-FATAL — they log warnings and sanitize, never crash.

Usage:
    from adk.safety import IntakeGuard, check_input, check_output

    guard = IntakeGuard()
    result = guard.check(user_message)
    if result.blocked:
        return "I can't process that request."

    # Or use the convenience functions
    safe_msg = check_input(user_message)  # Returns sanitized or original
    check_output(response)  # Logs warnings, returns cleaned response
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("adk.safety")


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """Result from a safety check."""
    safe: bool = True
    severity: Severity = Severity.NONE
    blocked: bool = False
    warnings: list[str] = field(default_factory=list)
    sanitized_content: str = ""
    patterns_matched: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Injection patterns (from AitherOS IntakeGuard)
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    # System prompt extraction
    (r"(?i)(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?", Severity.HIGH),
    (r"(?i)(?:repeat|show|reveal|print|output)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)", Severity.HIGH),
    (r"(?i)what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions?|rules?)", Severity.MEDIUM),

    # Role manipulation
    (r"(?i)you\s+are\s+(?:now|actually)\s+(?:a|an|the)\s+", Severity.MEDIUM),
    (r"(?i)(?:pretend|act|behave)\s+(?:as|like)\s+(?:you\s+are\s+)?(?:a|an|the)\s+", Severity.LOW),
    (r"(?i)from\s+now\s+on\s+(?:you|your)\s+(?:are|will|should|must)", Severity.MEDIUM),

    # Delimiter injection
    (r"(?i)\[/?(?:SYSTEM|INST|CONTEXT|USER|ASSISTANT)\]", Severity.HIGH),
    (r"(?i)<\|(?:im_start|im_end|system|user|assistant)\|>", Severity.HIGH),
    (r"(?i)```(?:system|instruction|prompt)", Severity.MEDIUM),

    # Dangerous tool invocations (preventing prompt-injected tool calls)
    (r"(?i)(?:execute|run|call)\s+(?:the\s+)?(?:tool|function|command)\s+['\"]?(?:rm|del|format|drop|delete)", Severity.CRITICAL),

    # Data exfiltration
    (r"(?i)(?:send|post|upload|exfil)\s+(?:all\s+)?(?:data|secrets?|keys?|tokens?|passwords?)\s+(?:to|via)", Severity.CRITICAL),
]

_COMPILED_PATTERNS = [(re.compile(p), s) for p, s in _INJECTION_PATTERNS]


class IntakeGuard:
    """Fast regex-based intake guard for prompt injection detection.

    Scans user messages for common injection patterns and returns
    a SafetyResult with severity assessment.
    """

    def __init__(self, block_threshold: Severity = Severity.HIGH):
        self.block_threshold = block_threshold
        self._severity_order = {
            Severity.NONE: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
            Severity.HIGH: 3, Severity.CRITICAL: 4,
        }

    def check(self, content: str) -> SafetyResult:
        """Check content for injection patterns.

        Returns SafetyResult with matched patterns and severity.
        """
        if not content:
            return SafetyResult()

        result = SafetyResult(sanitized_content=content)
        max_severity = Severity.NONE

        for pattern, severity in _COMPILED_PATTERNS:
            if pattern.search(content):
                result.patterns_matched.append(pattern.pattern[:80])
                result.warnings.append(f"{severity.value}: injection pattern detected")
                if self._severity_order[severity] > self._severity_order[max_severity]:
                    max_severity = severity

        result.severity = max_severity
        result.safe = max_severity == Severity.NONE

        # Block if severity meets threshold
        block_level = self._severity_order[self.block_threshold]
        if self._severity_order[max_severity] >= block_level:
            result.blocked = True
            result.sanitized_content = _sanitize(content)
            logger.warning(
                "IntakeGuard BLOCKED (severity=%s, patterns=%d): %s...",
                max_severity.value, len(result.patterns_matched),
                content[:100],
            )

        return result


def _sanitize(content: str) -> str:
    """Strip dangerous patterns from content."""
    sanitized = content
    for pattern, severity in _COMPILED_PATTERNS:
        if severity in (Severity.HIGH, Severity.CRITICAL):
            sanitized = pattern.sub("[FILTERED]", sanitized)
    return sanitized


# ─────────────────────────────────────────────────────────────────────────────
# Output validation
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_PATTERNS = [
    # Leaked secrets
    (r"(?i)(?:sk-[a-zA-Z0-9]{20,})", "Possible API key in output"),
    (r"(?i)(?:ghp_[a-zA-Z0-9]{30,})", "Possible GitHub token in output"),
    (r"(?i)(?:AKIA[A-Z0-9]{16})", "Possible AWS key in output"),
    # Internal prompt leakage
    (r"\[SYSTEM\]|\[AXIOMS\]|\[RULES\]|\[IDENTITY\]|\[CAPABILITIES\]|\[AFFECT\]", "System prompt leakage"),
    # Leaked tool_call XML tags (model exposed internal markup)
    (r"<tool_call>.*?</tool_call>", "Leaked tool_call XML in output"),
    (r"<tool_call>[^<]*$", "Truncated tool_call XML in output"),
]

_OUTPUT_COMPILED = [(re.compile(p), msg) for p, msg in _OUTPUT_PATTERNS]


def check_output(content: str) -> SafetyResult:
    """Check agent output for leaked secrets or system prompts."""
    result = SafetyResult(sanitized_content=content)
    for pattern, msg in _OUTPUT_COMPILED:
        if pattern.search(content):
            result.warnings.append(msg)
            result.safe = False
            result.severity = Severity.MEDIUM
            # Redact the match
            result.sanitized_content = pattern.sub("[REDACTED]", result.sanitized_content)
            logger.warning("Output safety: %s", msg)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

_guard: IntakeGuard | None = None


def check_input(content: str, block_threshold: Severity = Severity.HIGH) -> str:
    """Quick check — returns sanitized content if injection detected, else original.

    Convenience wrapper around IntakeGuard.check().
    """
    global _guard
    if _guard is None:
        _guard = IntakeGuard(block_threshold=block_threshold)

    result = _guard.check(content)
    if result.blocked:
        return result.sanitized_content
    return content
