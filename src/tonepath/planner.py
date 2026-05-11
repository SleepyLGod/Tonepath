"""Deterministic state path planning."""

from __future__ import annotations

import re

from tonepath.models import SessionPhase, SessionPlan, SessionRequest


DEFAULT_DURATION_SEC = 30 * 60


def parse_request(prompt: str) -> SessionRequest:
    """Parse a user prompt into a state-transition request."""

    duration_sec = parse_duration(prompt) or DEFAULT_DURATION_SEC
    source_state = infer_source_state(prompt)
    target_state = infer_target_state(prompt)
    no_vocals = any(token in prompt.lower() for token in ("no vocals", "instrumental")) or "不要人声" in prompt
    quiet = "quiet" in prompt.lower() or "安静" in prompt or "别太吵" in prompt
    return SessionRequest(
        prompt=prompt,
        source_state=source_state,
        target_state=target_state,
        duration_sec=duration_sec,
        no_vocals=no_vocals,
        quiet=quiet,
    )


def plan_session(prompt: str) -> SessionPlan:
    """Create a deterministic state-transition listening path."""

    request = parse_request(prompt)
    phases = build_phases(request)
    return SessionPlan(request=request, phases=tuple(phases))


def parse_duration(prompt: str) -> int | None:
    """Parse a duration from Chinese or English shorthand."""

    if "半小时" in prompt or "半个小时" in prompt:
        return 30 * 60
    match = re.search(r"(\d+)\s*(分钟|分|min|mins|minute|minutes|m)", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) * 60
    match = re.search(r"(\d+)\s*(小时|hour|hours|h)", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) * 3600
    return None


def infer_source_state(prompt: str) -> str:
    """Infer the user's starting state from the prompt."""

    if any(token in prompt for token in ("烦", "焦躁", "焦虑", "散", "乱")):
        return "irritated"
    if any(token in prompt for token in ("困", "累", "疲惫")):
        return "tired"
    if any(token in prompt for token in ("难过", "孤独", "emo")):
        return "low"
    return "unspecified"


def infer_target_state(prompt: str) -> str:
    """Infer the user's target state from the prompt."""

    if any(token in prompt for token in ("专注", "写代码", "学习", "工作", "论文")):
        return "focus"
    if any(token in prompt for token in ("睡", "放松", "平静")):
        return "calm"
    if any(token in prompt for token in ("精神", "启动", "清醒")):
        return "energized"
    return "steady"


def build_phases(request: SessionRequest) -> list[SessionPhase]:
    """Build phase targets for a listening path."""

    total = request.duration_sec
    first = max(total // 4, 60)
    second = max(total // 3, first + 60)
    second = min(second, total - 60)
    vocal_policy = "avoid" if request.no_vocals else "allow"

    if request.target_state == "focus":
        return [
            SessionPhase("decompress", 0, first, 0.35, 0.45, 0.35, vocal_policy),
            SessionPhase("stabilize", first, second, 0.45, 0.55, 0.45, vocal_policy),
            SessionPhase("focus", second, total, 0.5, 0.6, 0.5, vocal_policy),
        ]
    if request.target_state == "calm":
        return [
            SessionPhase("soften", 0, first, 0.3, 0.45, 0.3, vocal_policy),
            SessionPhase("settle", first, second, 0.25, 0.55, 0.25, vocal_policy),
            SessionPhase("calm", second, total, 0.2, 0.6, 0.2, vocal_policy),
        ]
    if request.target_state == "energized":
        return [
            SessionPhase("warmup", 0, first, 0.45, 0.5, 0.45, vocal_policy),
            SessionPhase("lift", first, second, 0.6, 0.6, 0.65, vocal_policy),
            SessionPhase("energize", second, total, 0.7, 0.65, 0.75, vocal_policy),
        ]
    return [
        SessionPhase("orient", 0, first, 0.4, 0.5, 0.4, vocal_policy),
        SessionPhase("steady", first, total, 0.45, 0.55, 0.45, vocal_policy),
    ]
