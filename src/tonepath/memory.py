"""Private memory logs, profile consolidation, and suggestion evidence."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import fcntl

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.llm import extract_chat_content, provider_config
from tonepath.profile import build_profile_evidence, sanitize_suggestions


LAST_CONSOLIDATED_SEQUENCE_KEY = "memory:last_consolidated_sequence"
MEMORY_LOG_LOCK = threading.Lock()
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
LAST_CONSOLIDATED_SEQUENCE_RE = re.compile(r"(?im)^#{1,6}\s*Last Consolidated Sequence:\s*\d+\s*$")
API_KEY_RE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{16,}|"
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|sk-proj-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{16,})\b"
)
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.-])/(?:Users|private|var|tmp|Volumes|opt|usr|home)/[^\s\"'`<>]+")
DEFAULT_MEMORY_PROFILE = "\n".join(
    [
        "# Tonepath Memory Profile",
        "",
        "_No consolidated memory profile yet._",
        "",
        "This file is for human-editable listening context. Tonepath may use it as LLM/Codex context, but it does not directly change recommendations.",
        "",
        "Run `uv run tonepath memory add \"...\"`, then `uv run tonepath memory consolidate --llm --confirm` to update it.",
        "",
    ]
)


def memory_root() -> Path:
    """Return the local private memory directory."""

    return config.ensure_data_dir() / "memory"


def memory_log_path() -> Path:
    """Return the append-only private memory log path."""

    return memory_root() / "logs" / "memory-log.jsonl"


def memory_profile_path() -> Path:
    """Return the human-editable consolidated memory profile path."""

    return memory_root() / "profile.md"


def memory_cache_dir(run_id: str) -> Path:
    """Return the cache directory for one memory run."""

    return config.ensure_data_dir() / "cache" / "memory" / run_id


def ensure_private_dir(path: Path) -> None:
    """Create a private owner-only directory for memory artifacts."""

    path.mkdir(parents=True, exist_ok=True)
    data_dir = config.ensure_data_dir()
    current = path
    while current != data_dir and current.is_relative_to(data_dir):
        os.chmod(current, PRIVATE_DIR_MODE)
        current = current.parent


def open_private_append(path: Path) -> TextIO:
    """Open a private owner-only file for appending text."""

    ensure_private_dir(path.parent)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, PRIVATE_FILE_MODE)
    os.chmod(path, PRIVATE_FILE_MODE)
    return os.fdopen(fd, "a", encoding="utf-8")


def write_private_text(path: Path, text: str) -> None:
    """Write text to a private owner-only file."""

    ensure_private_dir(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    finally:
        os.chmod(path, PRIVATE_FILE_MODE)


def add_memory_log(body: str, source: str = "cli") -> dict[str, object]:
    """Append one private memory note to the local JSONL log."""

    cleaned = body.strip()
    if not cleaned:
        raise ValueError("Memory body is empty.")
    path = memory_log_path()
    ensure_private_dir(path.parent)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with MEMORY_LOG_LOCK:
        with open_private_append(lock_path) as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                records = read_memory_logs()
                sequence = max((int(record.get("sequence", 0)) for record in records), default=0) + 1
                record = {
                    "id": f"mem-{sequence:06d}",
                    "sequence": sequence,
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "source": source,
                    "body": cleaned,
                }
                with open_private_append(path) as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return record


def read_memory_logs() -> list[dict[str, object]]:
    """Return all valid private memory log records."""

    path = memory_log_path()
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("body"), str):
            continue
        if "sequence" not in payload:
            continue
        try:
            payload["sequence"] = int(payload.get("sequence", 0))
        except (TypeError, ValueError):
            continue
        records.append(payload)
    return sorted(records, key=lambda record: int(record.get("sequence", 0)))


def memory_profile_text() -> str:
    """Return the consolidated memory profile Markdown or a guidance stub."""

    path = memory_profile_path()
    if not path.exists():
        return DEFAULT_MEMORY_PROFILE
    return path.read_text(encoding="utf-8")


def ensure_memory_profile() -> Path:
    """Create the memory profile file when it is missing and return its path."""

    path = memory_profile_path()
    ensure_private_dir(path.parent)
    if not path.exists():
        write_private_text(path, DEFAULT_MEMORY_PROFILE)
    return path


def last_consolidated_sequence(store: TonepathStore) -> int:
    """Return the last private memory log sequence that was consolidated."""

    value = store.get_app_state(LAST_CONSOLIDATED_SEQUENCE_KEY)
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0


def unconsolidated_memory_logs(store: TonepathStore) -> list[dict[str, object]]:
    """Return memory logs written after the current consolidation checkpoint."""

    checkpoint = last_consolidated_sequence(store)
    return [record for record in read_memory_logs() if int(record.get("sequence", 0)) > checkpoint]


def build_memory_evidence(store: TonepathStore) -> dict[str, object]:
    """Build a privacy-safe evidence pack for memory consolidation or suggestions."""

    logs = read_memory_logs()
    checkpoint = last_consolidated_sequence(store)
    new_logs = [record for record in logs if int(record.get("sequence", 0)) > checkpoint]
    run_id = uuid.uuid4().hex
    profile = memory_profile_text()
    redacted_profile, profile_privacy = redact_memory_text(profile)
    redacted_new_logs, new_logs_privacy = redact_memory_records(new_logs)
    redacted_recent_logs, recent_logs_privacy = redact_memory_records(logs[-20:])
    privacy = combine_memory_privacy(profile_privacy, new_logs_privacy, recent_logs_privacy)
    feedback_evidence = build_profile_evidence(store, limit=40)
    return {
        "run_id": run_id,
        "kind": "tonepath-memory-evidence",
        "privacy": {
            "contains_audio": False,
            "contains_absolute_paths": privacy["contains_absolute_paths"],
            "contains_api_keys": privacy["contains_api_keys"],
            "contains_full_library": False,
        },
        "summary": {
            "memory_logs": len(logs),
            "new_memory_logs": len(new_logs),
            "last_consolidated_sequence": checkpoint,
            "feedback_events": len(feedback_evidence.get("feedback_events", [])),
        },
        "memory_profile_markdown": redacted_profile,
        "new_memory_logs": public_memory_logs(redacted_new_logs),
        "recent_memory_logs": public_memory_logs(redacted_recent_logs),
        "profile_feedback_evidence": {
            "summary": feedback_evidence.get("summary", {}),
            "feedback_events": feedback_evidence.get("feedback_events", []),
        },
    }


def redact_memory_text(text: str) -> tuple[str, dict[str, bool]]:
    """Return redacted memory text and detected privacy flags."""

    contains_api_keys = bool(API_KEY_RE.search(text))
    contains_absolute_paths = bool(ABSOLUTE_PATH_RE.search(text))
    redacted = API_KEY_RE.sub(lambda match: f"{match.group(1)}[redacted-api-key]" if match.group(1) else "[redacted-api-key]", text)
    redacted = ABSOLUTE_PATH_RE.sub("[redacted-absolute-path]", redacted)
    return redacted, {"contains_api_keys": contains_api_keys, "contains_absolute_paths": contains_absolute_paths}


def redact_memory_records(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, bool]]:
    """Return memory records with body text redacted for evidence payloads."""

    redacted_records: list[dict[str, object]] = []
    contains_api_keys = False
    contains_absolute_paths = False
    for record in records:
        body, privacy = redact_memory_text(str(record.get("body", "")))
        contains_api_keys = contains_api_keys or privacy["contains_api_keys"]
        contains_absolute_paths = contains_absolute_paths or privacy["contains_absolute_paths"]
        redacted = dict(record)
        redacted["body"] = body
        redacted_records.append(redacted)
    return redacted_records, {"contains_api_keys": contains_api_keys, "contains_absolute_paths": contains_absolute_paths}


def combine_memory_privacy(*items: dict[str, bool]) -> dict[str, bool]:
    """Merge memory privacy flags from profile and log scans."""

    return {
        "contains_api_keys": any(item.get("contains_api_keys", False) for item in items),
        "contains_absolute_paths": any(item.get("contains_absolute_paths", False) for item in items),
    }


def public_memory_logs(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return memory logs with only context fields needed by LLM/Codex."""

    public: list[dict[str, object]] = []
    for record in records:
        public.append(
            {
                "id": str(record.get("id", "")),
                "sequence": int(record.get("sequence", 0)),
                "created_at": str(record.get("created_at", "")),
                "source": str(record.get("source", "cli")),
                "body": str(record.get("body", "")),
            }
        )
    return public


def write_memory_evidence(evidence: dict[str, object]) -> Path:
    """Write one memory evidence pack to the local cache."""

    run_id = str(evidence["run_id"])
    directory = memory_cache_dir(run_id)
    ensure_private_dir(directory)
    path = directory / "evidence.json"
    write_private_text(path, json.dumps(evidence, ensure_ascii=False, indent=2))
    return path


def consolidate_memory_with_llm(evidence: dict[str, object], provider: str | None = None) -> str:
    """Ask an opt-in LLM to consolidate private memory evidence into Markdown."""

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
                    "You consolidate Tonepath private memory logs into a human-editable Markdown profile. "
                    "The profile is listening context only, not therapy, diagnosis, or medical advice. "
                    "Do not invent music facts, BPM, vocalness, artists, genres, or audio evidence. "
                    "Preserve useful user-stated preferences and uncertainty. "
                    "Return JSON only with key profile_markdown."
                ),
            },
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        settings.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{settings.provider} memory consolidation failed.") from exc
    try:
        content = extract_chat_content(json.loads(body))
        parsed = json.loads(content)
    except (json.JSONDecodeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{settings.provider} memory consolidation returned an unparseable response.") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("profile_markdown"), str):
        raise RuntimeError("LLM memory consolidation must return profile_markdown.")
    return normalize_memory_profile_markdown(parsed["profile_markdown"])


def normalize_memory_profile_markdown(text: str) -> str:
    """Return a non-empty Tonepath memory profile Markdown document."""

    cleaned = text.strip()
    if not cleaned:
        raise RuntimeError("LLM memory consolidation returned an empty profile.")
    if not cleaned.lstrip().startswith("#"):
        cleaned = f"# Tonepath Memory Profile\n\n{cleaned}"
    return f"{cleaned.rstrip()}\n"


def memory_profile_with_checkpoint(text: str, sequence: int | None) -> str:
    """Return profile Markdown with a local, trustworthy consolidation checkpoint."""

    cleaned = normalize_memory_profile_markdown(text).rstrip()
    if sequence is None:
        return f"{cleaned}\n"
    marker = f"## Last Consolidated Sequence: {sequence}"
    if LAST_CONSOLIDATED_SEQUENCE_RE.search(cleaned):
        cleaned = LAST_CONSOLIDATED_SEQUENCE_RE.sub(marker, cleaned)
    else:
        cleaned = f"{cleaned}\n\n{marker}"
    return f"{cleaned.rstrip()}\n"


def save_consolidated_memory_profile(store: TonepathStore, evidence: dict[str, object], profile_markdown: str) -> Path:
    """Persist a consolidated memory profile and update the checkpoint."""

    path = memory_profile_path()
    ensure_private_dir(path.parent)
    sequences = [int(record.get("sequence", 0)) for record in evidence.get("new_memory_logs", []) if isinstance(record, dict)]
    last_sequence = max(sequences) if sequences else None
    write_private_text(path, memory_profile_with_checkpoint(profile_markdown, last_sequence))
    if sequences:
        store.set_app_state(LAST_CONSOLIDATED_SEQUENCE_KEY, str(last_sequence))
    return path


def memory_context_markdown(evidence: dict[str, object]) -> str:
    """Return Markdown memory context for profile suggestion prompts."""

    lines = [
        "# Tonepath Memory Context",
        "",
        "## Consolidated Profile",
        "",
        str(evidence.get("memory_profile_markdown") or DEFAULT_MEMORY_PROFILE).strip(),
        "",
        "## New Memory Logs",
        "",
    ]
    new_logs = [record for record in evidence.get("new_memory_logs", []) if isinstance(record, dict)]
    if not new_logs:
        lines.append("_No unconsolidated memory logs._")
    else:
        for record in new_logs[-20:]:
            lines.append(f"- {record.get('created_at', '--')} · {record.get('body', '')}")
    return "\n".join(lines).strip() + "\n"


def memory_suggestions_from_llm(evidence: dict[str, object], provider: str | None = None) -> list[dict[str, object]]:
    """Ask an opt-in LLM for profile suggestions from memory context and feedback evidence."""

    from tonepath.profile import suggest_with_llm

    profile_evidence = evidence.get("profile_feedback_evidence")
    if not isinstance(profile_evidence, dict):
        profile_evidence = {}
    suggestion_evidence = {
        "run_id": str(evidence.get("run_id") or uuid.uuid4().hex),
        "kind": "tonepath-memory-profile-suggestion-evidence",
        "privacy": evidence.get("privacy", {}),
        "summary": evidence.get("summary", {}),
        "feedback_events": profile_evidence.get("feedback_events", []),
    }
    suggestions = suggest_with_llm(suggestion_evidence, provider=provider, memory_context=memory_context_markdown(evidence))
    source = f"memory-{suggestions[0].get('source', 'llm')}" if suggestions else "memory-llm"
    for item in suggestions:
        item["source"] = source
    return sanitize_suggestions(suggestions, source=source)
