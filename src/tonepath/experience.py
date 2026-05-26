"""Smart default experience helpers for Tonepath."""

from __future__ import annotations

import shutil

from tonepath import config
from tonepath.llm import parse_prompt_with_llm, provider_config
from tonepath.models import SessionPlan, SessionRequest
from tonepath.planner import build_phases, plan_session


def setup_next_step(settings: config.TonepathConfig) -> str:
    """Return concise guidance after applying an experience preset."""

    if settings.experience.mode == "smart":
        return "Next: add a music directory, run `uv run tonepath prepare --full`, then `uv run tonepath listen \"...\"`."
    return "Next: add a music directory, run `uv run tonepath prepare`, then `uv run tonepath listen \"...\"`."


def listen_intelligence_summary(settings: config.TonepathConfig, runtime_ready: bool) -> str:
    """Return a compact product-mode summary for listen output."""

    model = "model ready" if runtime_ready else "model optional"
    try:
        provider = provider_config()
        llm = f"LLM {provider.provider} configured" if settings.privacy.send_to_llm and provider.configured else "LLM deterministic fallback"
    except ValueError:
        llm = "LLM deterministic fallback"
    codex = "Codex available" if shutil.which("codex") else "Codex optional"
    if settings.experience.mode == "private":
        return f"Intelligence: {model} · local deterministic · {codex}"
    return f"Intelligence: {model} · {llm} · {codex}"


def smart_plan_session(prompt: str, settings: config.TonepathConfig) -> tuple[SessionPlan, str | None]:
    """Return an LLM-assisted plan when Smart mode is enabled and configured."""

    if settings.experience.mode != "smart" or not settings.privacy.send_to_llm:
        return plan_session(prompt), None
    try:
        provider = provider_config()
    except ValueError:
        return plan_session(prompt), "LLM intent: provider config invalid; using deterministic parser."
    if not provider.configured:
        return plan_session(prompt), f"LLM intent: {provider.api_key_env} missing; using deterministic parser."
    try:
        parsed = parse_prompt_with_llm(prompt, provider=provider.provider)
    except RuntimeError as exc:
        exc_text = str(exc).strip() or "request failed"
        return plan_session(prompt), f"LLM intent: {exc_text}. Using deterministic parser."
    return plan_from_llm_payload(prompt, parsed), f"LLM intent: parsed with {provider.provider}."


def plan_from_llm_payload(prompt: str, payload: dict[str, object]) -> SessionPlan:
    """Build a safe session plan from a validated LLM intent payload."""

    fallback = plan_session(prompt)
    source_state = safe_state(payload.get("source_state"), {fallback.request.source_state, "unspecified", "tired", "irritated", "low"})
    target_state = safe_state(payload.get("target_state"), {"energized", "calm", "steady", "focus"})
    duration_min = safe_int(payload.get("duration_min"), fallback.request.duration_sec // 60)
    constraints = payload.get("constraints")
    constraint_labels = {str(item).lower() for item in constraints} if isinstance(constraints, list) else set()
    request = SessionRequest(
        prompt=prompt,
        source_state=source_state or fallback.request.source_state,
        target_state=target_state or fallback.request.target_state,
        duration_sec=max(1, duration_min) * 60,
        no_vocals=bool({"avoid_vocals", "no_vocals", "no vocals"} & constraint_labels),
        quiet=bool({"low_stimulation", "quiet", "low stimulation"} & constraint_labels),
    )
    return SessionPlan(request=request, phases=tuple(build_phases(request)))


def safe_state(value: object, allowed: set[str]) -> str | None:
    """Return an allowed state label from external input."""

    text = str(value or "").strip()
    return text if text in allowed else None


def safe_int(value: object, default: int) -> int:
    """Return an integer from external input, falling back on invalid values."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
