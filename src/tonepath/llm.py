"""Optional LLM helpers for prompt parsing and wording."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


PROVIDER_ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
}
PROVIDER_URLS = {
    "deepseek": "https://api.deepseek.com/chat/completions",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}
PROVIDER_MODELS = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
}


@dataclass(frozen=True)
class LlmProviderConfig:
    """Configuration for one OpenAI-compatible LLM provider."""

    provider: str
    api_key_env: str
    configured: bool
    model: str
    url: str


def active_provider() -> str:
    """Return the configured LLM provider name."""

    override = os.environ.get("TONEPATH_LLM_PROVIDER", "").strip()
    if override:
        from tonepath.config import normalize_llm_provider

        return normalize_llm_provider(override)

    from tonepath.config import load_config

    return load_config().llm.provider


def provider_config(provider: str | None = None) -> LlmProviderConfig:
    """Return sanitized provider configuration without exposing secrets."""

    name = (provider or active_provider()).strip().lower()
    if name not in PROVIDER_ENV_KEYS:
        raise ValueError("Only deepseek and qwen LLM providers are supported.")
    key_env = PROVIDER_ENV_KEYS[name]
    model = os.environ.get(f"TONEPATH_{name.upper()}_MODEL", PROVIDER_MODELS[name])
    url = os.environ.get(f"TONEPATH_{name.upper()}_BASE_URL", PROVIDER_URLS[name])
    return LlmProviderConfig(
        provider=name,
        api_key_env=key_env,
        configured=bool(os.environ.get(key_env)),
        model=model,
        url=url,
    )


def llm_doctor(provider: str | None = None) -> str:
    """Return a redacted LLM configuration report."""

    settings = provider_config(provider)
    return "\n".join(
        [
            "Tonepath LLM doctor",
            f"Provider: {settings.provider}",
            f"Model: {settings.model}",
            f"Endpoint: {settings.url}",
            f"API key env: {settings.api_key_env} ({'configured' if settings.configured else 'missing'})",
            "Secrets: not displayed",
        ]
    )


def parse_prompt_with_llm(prompt: str, provider: str | None = None) -> dict[str, Any]:
    """Parse one user prompt into state-transition intent using an opt-in LLM."""

    settings = provider_config(provider)
    api_key = os.environ.get(settings.api_key_env)
    if not api_key:
        raise RuntimeError(f"{settings.provider} requires {settings.api_key_env}.")
    payload = {
        "model": settings.model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Parse a music listening request into strict JSON only. "
                    "Do not infer audio facts, track facts, BPM, vocalness, genre, or artist metadata. "
                    "Allowed keys: source_state, target_state, duration_min, constraints."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        settings.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{settings.provider} request failed.") from exc
    content = extract_chat_content(json.loads(body))
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM returned a non-object intent.")
    return sanitize_intent(parsed)


def extract_chat_content(payload: dict[str, Any]) -> str:
    """Extract OpenAI-compatible assistant content from a chat response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not include choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("LLM response did not include message content.")
    return message["content"]


def sanitize_intent(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only the allowed intent keys with simple value types."""

    constraints = payload.get("constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    return {
        "source_state": string_or_default(payload.get("source_state"), "unspecified"),
        "target_state": string_or_default(payload.get("target_state"), "focus"),
        "duration_min": int_or_default(payload.get("duration_min"), 30),
        "constraints": [str(item) for item in constraints if isinstance(item, str)],
    }


def string_or_default(value: object, default: str) -> str:
    """Return a non-empty string value or a default."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def int_or_default(value: object, default: int) -> int:
    """Return a positive integer value or a default."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default
