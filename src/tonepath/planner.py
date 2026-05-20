"""Deterministic state path planning."""

from __future__ import annotations

import re
from dataclasses import replace

from tonepath.models import SessionPhase, SessionPlan, SessionRequest


DEFAULT_DURATION_SEC = 30 * 60
SOURCE_STATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tired": ("困", "累", "疲惫", "刚睡醒", "tired", "sleepy", "woke up", "just woke"),
    "irritated": ("烦", "焦躁", "焦虑", "散", "乱", "anxious", "stressed", "irritated", "scattered"),
    "low": ("难过", "孤独", "emo", "低落", "feel low", "feeling low", "lonely", "sad"),
}
TARGET_STATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "energized": (
        "提神",
        "醒脑",
        "启动",
        "清醒",
        "动力",
        "energize",
        "energizing",
        "wake up",
        "lift",
        "alert",
        "motivation",
        "start a task",
    ),
    "calm": (
        "冷静",
        "睡",
        "放松",
        "平静",
        "冥想",
        "calm down",
        "calmer",
        "calm",
        "relax",
        "relaxation",
        "sleep",
        "meditate",
    ),
    "steady": (
        "整理房间",
        "做家务",
        "整理资料",
        "有节奏",
        "稳定",
        "收尾",
        "clean my room",
        "chores",
        "organize files",
        "rhythmic",
        "steady",
        "wrap up",
    ),
    "focus": (
        "专注",
        "写代码",
        "学习",
        "工作",
        "论文",
        "深度工作",
        "阅读",
        "focus",
        "focused",
        "code",
        "study",
        "write a paper",
        "paper",
        "deep work",
        "read quietly",
    ),
}
NO_VOCALS_KEYWORDS = (
    "不要人声",
    "无人声",
    "纯音乐",
    "不要歌词",
    "no vocals",
    "no vocal",
    "instrumental",
    "no lyrics",
    "without vocals",
)
QUIET_KEYWORDS = (
    "安静",
    "低刺激",
    "别太吵",
    "不要太吵",
    "不太吵",
    "不吵",
    "quiet",
    "low stimulation",
    "low-stimulation",
    "not too loud",
)
CHINESE_DIGITS: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
ENGLISH_NUMBERS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def parse_request(prompt: str) -> SessionRequest:
    """Parse a user prompt into a state-transition request."""

    duration_sec = parse_duration(prompt) or DEFAULT_DURATION_SEC
    source_state = infer_source_state(prompt)
    target_state = infer_target_state(prompt)
    no_vocals = contains_any(prompt, NO_VOCALS_KEYWORDS)
    quiet = contains_any(prompt, QUIET_KEYWORDS)
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


def request_constraints(request: SessionRequest) -> list[str]:
    """Return stable public constraint labels for a parsed request."""

    constraints: list[str] = []
    if request.no_vocals:
        constraints.append("avoid_vocals")
    if request.quiet:
        constraints.append("low_stimulation")
    return constraints


def parse_duration(prompt: str) -> int | None:
    """Parse a duration from Chinese or English shorthand."""

    normalized = prompt.lower()
    if any(token in normalized for token in ("半小时", "半个小时", "half an hour", "half hour")):
        return 30 * 60
    match = re.search(r"(\d+)\s*(分钟|分|min|mins|minute|minutes|m)", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) * 60
    match = re.search(r"(\d+)\s*(小时|hour|hours|h)", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) * 3600
    match = re.search(r"([零一二两三四五六七八九十]+)\s*(分钟|分)", prompt)
    if match:
        return parse_chinese_number(match.group(1)) * 60
    match = re.search(r"([零一二两三四五六七八九十]+)\s*(小时)", prompt)
    if match:
        return parse_chinese_number(match.group(1)) * 3600
    number_words = "|".join(sorted(ENGLISH_NUMBERS, key=len, reverse=True))
    match = re.search(rf"\b({number_words})(?:[- ]({number_words}))?\s+(minutes?|mins?|hours?|hrs?)\b", normalized)
    if match:
        value = parse_english_number(" ".join(part for part in match.group(1, 2) if part))
        unit = match.group(3)
        if value is not None:
            return value * (3600 if unit.startswith(("hour", "hr")) else 60)
    return None


def infer_source_state(prompt: str) -> str:
    """Infer the user's starting state from the prompt."""

    for state, keywords in SOURCE_STATE_KEYWORDS.items():
        if contains_any(prompt, keywords):
            return state
    return "unspecified"


def infer_target_state(prompt: str) -> str:
    """Infer the user's target state from the prompt."""

    for state in ("energized", "calm", "steady", "focus"):
        if contains_any(prompt, TARGET_STATE_KEYWORDS[state]):
            return state
    return "steady"


def contains_any(prompt: str, keywords: tuple[str, ...]) -> bool:
    """Return whether a prompt contains any keyword in a case-insensitive way."""

    normalized = prompt.lower()
    return any(keyword.lower() in normalized for keyword in keywords)


def parse_chinese_number(value: str) -> int:
    """Parse a small Chinese integer used in duration phrases."""

    if value == "十":
        return 10
    if "十" in value:
        tens, _, ones = value.partition("十")
        tens_value = CHINESE_DIGITS.get(tens, 1) if tens else 1
        ones_value = CHINESE_DIGITS.get(ones, 0) if ones else 0
        return tens_value * 10 + ones_value
    return sum(CHINESE_DIGITS[char] for char in value)


def parse_english_number(value: str) -> int | None:
    """Parse a small English integer used in duration phrases."""

    normalized = value.lower().replace("-", " ")
    if normalized in ENGLISH_NUMBERS:
        return ENGLISH_NUMBERS[normalized]
    parts = normalized.split()
    if len(parts) == 2 and parts[0] in ENGLISH_NUMBERS and parts[1] in ENGLISH_NUMBERS:
        tens = ENGLISH_NUMBERS[parts[0]]
        ones = ENGLISH_NUMBERS[parts[1]]
        if tens >= 20 and ones < 10:
            return tens + ones
    return None


def build_phases(request: SessionRequest) -> list[SessionPhase]:
    """Build phase targets for a listening path."""

    total = request.duration_sec
    first = max(total // 4, 60)
    second = max(total // 3, first + 60)
    second = min(second, total - 60)
    vocal_policy = "avoid" if request.no_vocals else "allow"

    if request.target_state == "focus":
        return quiet_adjusted_phases(request, [
            SessionPhase("decompress", 0, first, 0.35, 0.45, 0.35, vocal_policy),
            SessionPhase("stabilize", first, second, 0.45, 0.55, 0.45, vocal_policy),
            SessionPhase("focus", second, total, 0.5, 0.6, 0.5, vocal_policy),
        ])
    if request.target_state == "calm":
        return quiet_adjusted_phases(request, [
            SessionPhase("soften", 0, first, 0.3, 0.45, 0.3, vocal_policy),
            SessionPhase("settle", first, second, 0.25, 0.55, 0.25, vocal_policy),
            SessionPhase("calm", second, total, 0.2, 0.6, 0.2, vocal_policy),
        ])
    if request.target_state == "energized":
        return [
            SessionPhase("warmup", 0, first, 0.45, 0.5, 0.45, vocal_policy),
            SessionPhase("lift", first, second, 0.6, 0.6, 0.65, vocal_policy),
            SessionPhase("energize", second, total, 0.7, 0.65, 0.75, vocal_policy),
        ]
    return quiet_adjusted_phases(request, [
        SessionPhase("orient", 0, first, 0.4, 0.5, 0.4, vocal_policy),
        SessionPhase("steady", first, total, 0.45, 0.55, 0.45, vocal_policy),
    ])


def quiet_adjusted_phases(request: SessionRequest, phases: list[SessionPhase]) -> list[SessionPhase]:
    """Lower phase targets when the user asks for quiet or low stimulation."""

    if not request.quiet:
        return phases
    adjusted: list[SessionPhase] = []
    for phase in phases:
        if phase.label in {"soften", "settle", "calm"}:
            adjusted.append(lower_phase_targets(phase, amount=0.08))
        elif phase.label in {"decompress", "focus"}:
            adjusted.append(lower_phase_targets(phase, amount=0.08))
        elif phase.label == "stabilize":
            adjusted.append(lower_phase_targets(phase, amount=0.06))
        elif phase.label in {"orient", "steady"}:
            adjusted.append(lower_phase_targets(phase, amount=0.06))
        else:
            adjusted.append(phase)
    return adjusted


def lower_phase_targets(phase: SessionPhase, amount: float) -> SessionPhase:
    """Return a phase with lower arousal and energy targets."""

    return replace(
        phase,
        target_arousal=max(phase.target_arousal - amount, 0.1),
        target_energy=max(phase.target_energy - amount, 0.1),
    )
