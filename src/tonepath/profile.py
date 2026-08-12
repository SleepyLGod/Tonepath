"""Profile evidence, LLM suggestions, and local rule application."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import uuid
from importlib import resources
from pathlib import Path
from typing import Any

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.display import display_artist, display_label, display_title
from tonepath.llm import extract_chat_content, provider_config
from tonepath.models import ProfileRule


SUPPORTED_RULE_TYPES = {
    "prefer_lower_loudness",
    "prefer_lower_energy",
    "prefer_lower_vocalness",
    "demote_high_bpm",
    "prefer_artist",
}

GENERATED_START = "<!-- tonepath:generated:start -->"
GENERATED_END = "<!-- tonepath:generated:end -->"
HUMAN_NOTES_START = "<!-- tonepath:human-notes:start -->"
HUMAN_NOTES_END = "<!-- tonepath:human-notes:end -->"
DEFAULT_HUMAN_NOTES = "Add personal listening notes here. Tonepath preserves this section when regenerating the file."


def build_profile_evidence(store: TonepathStore, limit: int = 80) -> dict[str, object]:
    """Build a privacy-safe evidence pack for profile suggestion."""

    rows = store.conn.execute(
        """
        SELECT
          feedback.id AS feedback_id,
          feedback.type AS feedback_type,
          feedback.created_at AS feedback_at,
          sessions.prompt,
          sessions.source_state,
          sessions.target_state,
          tracks.id AS track_id,
          tracks.title,
          tracks.artist,
          tracks.album,
          tracks.genre,
          tracks.duration,
          track_features.bpm,
          track_features.loudness,
          track_features.energy,
          track_features.vocalness,
          track_features.feature_source,
          track_features.confidence AS feature_confidence
        FROM feedback
        LEFT JOIN sessions ON sessions.id = feedback.session_id
        LEFT JOIN tracks ON tracks.id = feedback.track_id
        LEFT JOIN track_features ON track_features.track_id = tracks.id
        ORDER BY feedback.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    events: list[dict[str, object]] = []
    for row in rows:
        track = None
        if row["track_id"] is not None:
            title = row["title"]
            artist = row["artist"]
            track = {
                "id": int(row["track_id"]),
                "title": display_title_from_values(title, row["album"]),
                "artist": display_artist_from_values(artist),
                "label": display_label_from_values(title, artist, row["album"]),
                "genre": row["genre"],
                "duration": row["duration"],
                "features": {
                    "bpm": row["bpm"],
                    "loudness": row["loudness"],
                    "energy": row["energy"],
                    "vocalness": row["vocalness"],
                    "source": row["feature_source"],
                    "confidence": row["feature_confidence"],
                },
            }
        events.append(
            {
                "feedback_id": int(row["feedback_id"]),
                "feedback_type": str(row["feedback_type"]),
                "created_at": str(row["feedback_at"]),
                "prompt": row["prompt"],
                "source_state": row["source_state"],
                "target_state": row["target_state"],
                "track": track,
            }
        )
    run_id = uuid.uuid4().hex
    reactions = profile_reactions(store)
    return {
        "run_id": run_id,
        "kind": "tonepath-profile-evidence",
        "privacy": {
            "contains_audio": False,
            "contains_absolute_paths": False,
            "contains_api_keys": False,
            "contains_full_library": False,
        },
        "summary": store.profile_summary(),
        "current_reactions_are_authoritative": True,
        "track_reactions": reactions,
        "feedback_events": list(reversed(events)),
    }


def profile_reactions(store: TonepathStore) -> list[dict[str, object]]:
    """Return current reactions with privacy-safe track evidence."""

    reactions: list[dict[str, object]] = []
    for row in store.list_track_reactions():
        track_id = int(row["track_id"])
        track = store.get_track(track_id, effective_metadata=True)
        if track is None:
            continue
        features = store.get_features(track_id)
        reactions.append(
            {
                "track_id": track_id,
                "reaction": row["reaction"],
                "title": display_title(track),
                "artist": display_artist(track),
                "label": display_label(track),
                "genre": track.genre,
                "duration": track.duration,
                "updated_at": row["updated_at"],
                "features": {
                    "bpm": features.bpm if features is not None else None,
                    "loudness": features.loudness if features is not None else None,
                    "energy": features.energy if features is not None else None,
                    "vocalness": features.vocalness if features is not None else None,
                    "source": features.feature_source if features is not None else None,
                    "confidence": features.confidence if features is not None else None,
                },
            }
        )
    return reactions


def write_profile_evidence(evidence: dict[str, object]) -> Path:
    """Write one profile evidence pack to the local cache."""

    run_id = str(evidence["run_id"])
    directory = profile_cache_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_profile_memory(store: TonepathStore) -> Path:
    """Write a human-editable Markdown profile memory file."""

    path = profile_memory_path()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    human_notes = extract_human_notes(existing)
    evidence = build_profile_evidence(store, limit=20)
    rules = store.list_profile_rules()
    generated = "\n".join(
        [
            "## Profile Status",
            "",
            "This file is editable. Tonepath uses it as LLM/Codex context only; selector behavior changes only after a validated suggestion is applied.",
            "",
            "## Local Summary",
            "",
            markdown_kv_table(evidence.get("summary")),
            "",
            "## Active Rules",
            "",
            markdown_rule_table(rules),
            "",
            "## Pending Suggestions",
            "",
            markdown_suggestion_table(list_pending_suggestions()),
            "",
            "## Recent Feedback Evidence",
            "",
            markdown_feedback_table(evidence),
            "",
            "## Current Track Reactions",
            "",
            markdown_reaction_table(evidence),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_document("Tonepath Profile Memory", generated, human_notes), encoding="utf-8")
    return path


def write_profile_evidence_markdown(
    evidence: dict[str, object], rules: list[ProfileRule] | None = None, pending_suggestions: list[dict[str, object]] | None = None
) -> Path:
    """Write a human-readable Markdown evidence snapshot."""

    run_id = str(evidence["run_id"])
    latest_path = profile_evidence_latest_path()
    existing = latest_path.read_text(encoding="utf-8") if latest_path.exists() else ""
    human_notes = extract_human_notes(existing)
    generated = "\n".join(
        [
            "## Privacy Boundary",
            "",
            markdown_kv_table(evidence.get("privacy")),
            "",
            "## Local Summary",
            "",
            markdown_kv_table(evidence.get("summary")),
            "",
            "## Active Rules",
            "",
            markdown_rule_table(rules or []),
            "",
            "## Pending Suggestions",
            "",
            markdown_suggestion_table(pending_suggestions or []),
            "",
            "## Recent Feedback Events",
            "",
            markdown_feedback_table(evidence),
            "",
            "## Current Track Reactions",
            "",
            markdown_reaction_table(evidence),
            "",
            "## LLM Instructions",
            "",
            "- Suggest profile rules only when supported by the evidence above or explicit human notes.",
            "- Do not invent BPM, vocalness, genre, mood, artist facts, or track facts.",
            "- Return structured suggestions; Markdown is context, not an executable rule.",
        ]
    )
    text = markdown_document("Tonepath Profile Evidence", generated, human_notes)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(text, encoding="utf-8")
    archive_path = profile_evidence_dir() / f"{run_id}.md"
    archive_path.write_text(text, encoding="utf-8")
    return latest_path


def profile_data_dir() -> Path:
    """Return the local profile Markdown directory."""

    return config.ensure_data_dir() / "profile"


def profile_memory_path() -> Path:
    """Return the human-editable profile memory path."""

    return profile_data_dir() / "memory.md"


def profile_evidence_dir() -> Path:
    """Return the profile Markdown evidence directory."""

    return profile_data_dir() / "evidence"


def profile_evidence_latest_path() -> Path:
    """Return the latest profile Markdown evidence path."""

    return profile_evidence_dir() / "latest.md"


def profile_cache_dir(run_id: str) -> Path:
    """Return the cache directory for one profile suggestion run."""

    return config.ensure_data_dir() / "cache" / "profile" / run_id


def deterministic_suggestions(evidence: dict[str, object]) -> list[dict[str, object]]:
    """Generate conservative profile suggestions from local feedback only."""

    events = [event for event in evidence.get("feedback_events", []) if isinstance(event, dict)]
    suggestions: list[dict[str, object]] = []
    focus_loud = [
        event
        for event in events
        if event.get("target_state") == "focus"
        and event.get("feedback_type") == "too-loud"
        and feature_value(event, "loudness") is not None
    ]
    if len(focus_loud) >= 1:
        suggestions.append(
            suggestion(
                "focus-lower-loudness",
                "focus",
                "prefer_lower_loudness",
                "loudness",
                threshold=-12.0,
                weight=0.7,
                confidence="medium",
                source="deterministic-profile",
                rationale="Focus prefers quieter tracks after too-loud feedback in focus sessions.",
                evidence_count=len(focus_loud),
            )
        )

    reactions = [reaction for reaction in evidence.get("track_reactions", []) if isinstance(reaction, dict)]
    liked_low_vocal = [
        reaction
        for reaction in reactions
        if reaction.get("reaction") == "liked"
        and reaction_feature_value(reaction, "vocalness") is not None
        and float(reaction_feature_value(reaction, "vocalness")) <= 0.35
    ]
    no_vocals = [event for event in events if event.get("feedback_type") == "no-vocals"]
    if liked_low_vocal or no_vocals:
        suggestions.append(
            suggestion(
                "global-lower-vocalness",
                "global",
                "prefer_lower_vocalness",
                "vocalness",
                threshold=0.35,
                weight=0.6,
                confidence="medium" if liked_low_vocal else "low",
                source="deterministic-profile",
                rationale="Feedback suggests a preference for lower-vocalness tracks.",
                evidence_count=len(liked_low_vocal) + len(no_vocals),
            )
        )

    skipped_fast_focus = [
        event
        for event in events
        if event.get("target_state") == "focus"
        and event.get("feedback_type") == "skip"
        and (feature_value(event, "bpm") is not None and float(feature_value(event, "bpm")) >= 135.0)
    ]
    if skipped_fast_focus:
        suggestions.append(
            suggestion(
                "focus-demote-high-bpm",
                "focus",
                "demote_high_bpm",
                "bpm",
                threshold=135.0,
                weight=0.8,
                confidence="medium",
                source="deterministic-profile",
                rationale="Skipped focus tracks include high BPM; demote similar high-BPM focus candidates.",
                evidence_count=len(skipped_fast_focus),
            )
        )
    return suggestions


def suggest_with_llm(evidence: dict[str, object], provider: str | None = None, memory_context: str | None = None) -> list[dict[str, object]]:
    """Ask an opt-in LLM for profile suggestions from a safe evidence pack."""

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
                    "You summarize Tonepath user listening feedback into strict JSON only. "
                    "Do not invent audio facts, track facts, genre, mood, BPM, or vocalness. "
                    "Use only the provided evidence. Output an object with a suggestions array. "
                    "If focus feedback includes too-loud feedback and known loudness, suggest prefer_lower_loudness. "
                    "If a no-vocals or focus context includes a liked low-vocalness track, suggest prefer_lower_vocalness. "
                    "If focus feedback skips tracks with BPM >= 135, suggest demote_high_bpm. "
                    "Weak evidence may produce low-confidence suggestions, but never invent missing fields. "
                    "Each suggestion must include suggestion_id, scope, rule_type, target, threshold, weight, "
                    "confidence, rationale, and evidence_count. Supported rule_type values are: "
                    "prefer_lower_loudness, prefer_lower_energy, prefer_lower_vocalness, demote_high_bpm, prefer_artist."
                ),
            },
            {"role": "user", "content": llm_profile_payload(evidence, memory_context)},
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
        raise RuntimeError(f"{settings.provider} profile suggestion failed.") from exc
    content = extract_chat_content(json.loads(body))
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM profile suggestion must be a JSON object.")
    return sanitize_suggestions(parsed.get("suggestions"), source=f"llm-{settings.provider}")


def run_codex_profile_suggest(evidence_path: Path, web: bool = False, memory_paths: list[Path] | None = None) -> dict[str, object]:
    """Run Codex against one profile evidence pack."""

    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("Codex CLI is not available on PATH. Install Codex or rerun without --codex.")
    result_path = evidence_path.parent / "codex-result.json"
    command = [codex]
    if web:
        command.append("--search")
    command.extend(
        [
            "exec",
            "--sandbox",
            "read-only",
            "--cd",
            str(config.repo_root()),
            "--output-schema",
            str(profile_schema_path()),
            "-o",
            str(result_path),
            "-",
        ]
    )
    subprocess.run(
        command,
        input=profile_codex_prompt(evidence_path, web=web, memory_paths=memory_paths),
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    suggestions = sanitize_suggestions(payload.get("suggestions"), source="codex-profile")
    payload["suggestions"] = suggestions
    return payload


def save_suggestions(evidence: dict[str, object], suggestions: list[dict[str, object]], source: str) -> Path:
    """Save pending profile suggestions without applying them."""

    run_id = str(evidence["run_id"])
    directory = profile_cache_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "suggestions.json"
    payload = {"run_id": run_id, "source": source, "suggestions": suggestions}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_suggestion(store: TonepathStore, suggestion_id: str) -> ProfileRule:
    """Apply one pending suggestion as a local profile rule."""

    suggestion_payload = find_pending_suggestion(suggestion_id)
    if suggestion_payload is None:
        raise RuntimeError(f"No pending profile suggestion found for {suggestion_id}.")
    rule = rule_from_suggestion(suggestion_payload)
    store.upsert_profile_rule(rule)
    return rule


def apply_suggestion_group(store: TonepathStore, group_id: str) -> dict[str, object]:
    """Apply every suggestion in one derived profile suggestion group."""

    group = find_suggestion_group(group_id)
    if group is None:
        raise RuntimeError(f"No profile suggestion group found for {group_id}.")
    active_keys = {rule.key for rule in store.list_profile_rules()}
    applied: list[str] = []
    skipped: list[str] = []
    records = group.get("_records")
    if not isinstance(records, list):
        raise RuntimeError(f"Profile suggestion group {group_id} is malformed.")
    for record in records:
        if not isinstance(record, dict):
            continue
        suggestion_id = str(record.get("suggestion_id", "--"))
        rule_key = str(record.get("_rule_key") or "")
        if rule_key in active_keys:
            skipped.append(suggestion_id)
            continue
        rule = rule_from_suggestion(record)
        store.upsert_profile_rule(rule)
        active_keys.add(rule.key)
        applied.append(suggestion_id)
    return {
        "group_id": group_id,
        "scope": group.get("scope", "global"),
        "applied": applied,
        "skipped": skipped,
    }


def find_pending_suggestion(suggestion_id: str) -> dict[str, object] | None:
    """Find one pending suggestion in the local profile cache."""

    for suggestion_item in suggestion_cache_records(include_applied=False):
        if suggestion_item.get("suggestion_id") == suggestion_id:
            return public_suggestion(suggestion_item)
    return None


def list_pending_suggestions() -> list[dict[str, object]]:
    """Return unapplied profile suggestions from the local cache."""

    return [public_suggestion(item) for item in suggestion_cache_records(include_applied=False)]


def pending_suggestion_groups() -> list[dict[str, object]]:
    """Return unapplied profile suggestions grouped for safer review."""

    return public_suggestion_groups(include_applied=False)


def find_suggestion_group(group_id: str) -> dict[str, object] | None:
    """Find one suggestion group, including already-applied suggestions."""

    for group in suggestion_groups(include_applied=True):
        if group.get("group_id") == group_id:
            return group
    return None


def public_suggestion_groups(include_applied: bool) -> list[dict[str, object]]:
    """Return suggestion groups without internal record payloads."""

    public: list[dict[str, object]] = []
    for group in suggestion_groups(include_applied=include_applied):
        public.append({key: value for key, value in group.items() if key != "_records"})
    return public


def suggestion_groups(include_applied: bool) -> list[dict[str, object]]:
    """Group profile suggestions by suggestion run and scope."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in suggestion_cache_records(include_applied=include_applied):
        run_id = str(record.get("_run_id") or "")
        scope = str(record.get("scope") or "global")
        grouped.setdefault((run_id, scope), []).append(record)

    groups: list[dict[str, object]] = []
    used_ids: set[str] = set()
    for (run_id, scope), records in grouped.items():
        base_id = profile_group_id(scope)
        group_id = unique_group_id(base_id, used_ids)
        used_ids.add(group_id)
        rule_types = [str(item.get("rule_type", "--")) for item in records]
        rationale_parts = [str(item.get("rationale", "")).strip() for item in records if str(item.get("rationale", "")).strip()]
        groups.append(
            {
                "group_id": group_id,
                "run_id": run_id,
                "scope": scope,
                "source": str(records[0].get("_source") or "--"),
                "suggestion_ids": [str(item.get("suggestion_id", "--")) for item in records],
                "rules": rule_types,
                "confidence": group_confidence(records),
                "evidence_count": sum(int_or_default(item.get("evidence_count"), 0) for item in records),
                "rationale": " ".join(rationale_parts[:2]),
                "hint": group_hint(set(rule_types), scope),
                "apply_command": f"uv run tonepath profile apply-group {group_id}",
                "_records": records,
            }
        )
    return groups


def suggestion_cache_records(include_applied: bool) -> list[dict[str, object]]:
    """Return normalized suggestion cache records with run metadata."""

    root = config.ensure_data_dir() / "cache" / "profile"
    if not root.exists():
        return []
    applied_keys = applied_profile_rule_keys()
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("*/suggestions.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        suggestions = payload.get("suggestions")
        if not isinstance(suggestions, list):
            continue
        for suggestion_item in suggestions:
            if not isinstance(suggestion_item, dict):
                continue
            key = suggestion_rule_key(suggestion_item)
            if key is None or (not include_applied and key in applied_keys):
                continue
            clean = sanitize_suggestion(suggestion_item, source=str(suggestion_item.get("source") or payload.get("source") or "profile-suggestion"))
            clean["_run_id"] = str(payload.get("run_id") or path.parent.name)
            clean["_source"] = str(payload.get("source") or clean.get("source") or "profile-suggestion")
            clean["_rule_key"] = key
            clean["_mtime"] = path.stat().st_mtime
            records.append(clean)
    return records


def public_suggestion(record: dict[str, object]) -> dict[str, object]:
    """Return a suggestion payload without internal cache metadata."""

    return {key: value for key, value in record.items() if not key.startswith("_")}


def profile_group_id(scope: str) -> str:
    """Return a stable readable suggestion group id for a scope."""

    slug = re.sub(r"[^a-z0-9]+", "-", scope.lower()).strip("-") or "global"
    return f"{slug}-profile-group"


def unique_group_id(base_id: str, used_ids: set[str]) -> str:
    """Return a non-conflicting profile group id."""

    if base_id not in used_ids:
        return base_id
    index = 2
    while f"{base_id}-{index}" in used_ids:
        index += 1
    return f"{base_id}-{index}"


def group_confidence(records: list[dict[str, object]]) -> str:
    """Return the strongest confidence label in a suggestion group."""

    rank = {"low": 0, "medium": 1, "high": 2}
    best = "low"
    for record in records:
        confidence = str(record.get("confidence") or "low")
        if rank.get(confidence, 0) > rank[best]:
            best = confidence
    return best


def group_hint(rule_types: set[str], scope: str) -> str:
    """Return a short safety hint for complementary profile rules."""

    if {"prefer_lower_vocalness", "demote_high_bpm"}.issubset(rule_types):
        return f"Apply together for low-vocal / low-stimulation {scope} preferences."
    return ""


def applied_profile_rule_keys() -> set[str]:
    """Return keys for active profile rules."""

    store = TonepathStore()
    try:
        return {rule.key for rule in store.list_profile_rules()}
    finally:
        store.close()


def suggestion_rule_key(payload: dict[str, object]) -> str | None:
    """Return the profile rule key implied by one suggestion."""

    try:
        clean = sanitize_suggestion(payload, source=str(payload.get("source") or "profile-suggestion"))
    except RuntimeError:
        return None
    return f"{clean['scope']}:{clean['rule_type']}:{clean['target']}"


def active_rule_payload(rule: ProfileRule) -> dict[str, object]:
    """Return one active profile rule as a display-safe payload."""

    parse_error = False
    try:
        payload = json.loads(rule.value)
    except (json.JSONDecodeError, TypeError):
        payload = {}
        parse_error = True
    return {
        "id": rule.id,
        "key": rule.key,
        "scope": str(payload.get("scope", "global")),
        "rule_type": str(payload.get("rule_type", "unknown")),
        "target": str(payload.get("target", "unknown")),
        "threshold": payload.get("threshold"),
        "weight": payload.get("weight"),
        "source": rule.source,
        "confidence": rule.confidence,
        "rationale": str(payload.get("rationale", "")),
        "evidence_count": payload.get("evidence_count"),
        "parse_error": parse_error,
    }


def profile_readiness(summary: dict[str, int], active_rules: list[dict[str, object]], pending_suggestions: list[dict[str, object]]) -> str:
    """Return a user-facing profile learning readiness state."""

    if active_rules:
        return "Active profile rules"
    if pending_suggestions:
        return "Suggestions pending"
    if summary.get("feedback", 0) == 0 and summary.get("track_reactions", 0) == 0:
        return "No feedback yet"
    return "Feedback ready for suggestions"


def profile_learning_hint() -> str:
    """Return a lightweight post-feedback profile learning hint."""

    if list_pending_suggestions():
        return "Run `uv run tonepath profile inspect` to review pending preference suggestions."
    return "Run `uv run tonepath profile suggest` to review possible preference rules."


def delete_profile_markdown() -> None:
    """Delete local human-editable profile Markdown files."""

    root = profile_data_dir()
    if root.exists():
        shutil.rmtree(root)


def delete_profile_cache() -> None:
    """Delete local pending profile suggestion cache files."""

    root = config.ensure_data_dir() / "cache" / "profile"
    if root.exists():
        shutil.rmtree(root)


def rule_from_suggestion(payload: dict[str, object]) -> ProfileRule:
    """Convert one validated suggestion to a stored profile rule."""

    clean = sanitize_suggestion(payload, source=str(payload.get("source") or "profile-suggestion"))
    key = f"{clean['scope']}:{clean['rule_type']}:{clean['target']}"
    return ProfileRule(
        id=None,
        key=key,
        value=json.dumps(clean, ensure_ascii=False, sort_keys=True),
        source=str(clean["source"]),
        confidence=str(clean["confidence"]),
    )


def sanitize_suggestions(value: object, source: str) -> list[dict[str, object]]:
    """Validate and normalize suggestion payloads."""

    if not isinstance(value, list):
        raise RuntimeError("Profile suggestions must be a list.")
    return [sanitize_suggestion(item, source=source) for item in value if isinstance(item, dict)]


def sanitize_suggestion(item: dict[str, object], source: str) -> dict[str, object]:
    """Validate and normalize one profile suggestion."""

    rule_type = str(item.get("rule_type") or "")
    if rule_type not in SUPPORTED_RULE_TYPES:
        raise RuntimeError(f"Unsupported profile rule type: {rule_type}")
    suggestion_id = str(item.get("suggestion_id") or item.get("id") or f"{item.get('scope', 'global')}-{rule_type}")
    scope = str(item.get("scope") or "global")
    target = str(item.get("target") or default_target(rule_type))
    confidence = str(item.get("confidence") or "low")
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    return {
        "suggestion_id": suggestion_id,
        "scope": scope,
        "rule_type": rule_type,
        "target": target,
        "threshold": float_or_default(item.get("threshold"), default_threshold(rule_type)),
        "weight": float_or_default(item.get("weight"), 0.5),
        "confidence": confidence,
        "source": str(item.get("source") or source),
        "rationale": str(item.get("rationale") or item.get("reason") or "Profile suggestion from local evidence."),
        "evidence_count": int_or_default(item.get("evidence_count"), 1),
    }


def suggestion(
    suggestion_id: str,
    scope: str,
    rule_type: str,
    target: str,
    *,
    threshold: float,
    weight: float,
    confidence: str,
    source: str,
    rationale: str,
    evidence_count: int,
) -> dict[str, object]:
    """Return one normalized deterministic suggestion."""

    return sanitize_suggestion(
        {
            "suggestion_id": suggestion_id,
            "scope": scope,
            "rule_type": rule_type,
            "target": target,
            "threshold": threshold,
            "weight": weight,
            "confidence": confidence,
            "source": source,
            "rationale": rationale,
            "evidence_count": evidence_count,
        },
        source=source,
    )


def llm_profile_payload(evidence: dict[str, object], memory_context: str | None) -> str:
    """Return the user message payload for LLM profile suggestion."""

    if memory_context is None:
        return json.dumps(evidence, ensure_ascii=False)
    return json.dumps(
        {
            "profile_evidence_json": evidence,
            "profile_memory_markdown": memory_context,
        },
        ensure_ascii=False,
    )


def profile_codex_prompt(evidence_path: Path, web: bool, memory_paths: list[Path] | None = None) -> str:
    """Return the prompt passed to Codex for profile suggestion."""

    web_line = "Web search is allowed only for public music context." if web else "Do not use web search."
    memory_lines = [f"Markdown context path: {path}" for path in memory_paths or []]
    return "\n".join(
        [
            "<task>",
            "Suggest Tonepath profile rules from the local profile evidence pack.",
            f"Skill path: {profile_skill_path()}",
            f"Evidence pack path: {evidence_path}",
            *memory_lines,
            web_line,
            "</task>",
            "<output_contract>",
            "Return JSON matching the provided output schema only.",
            "Do not modify files, SQLite, playback state, or profile rules.",
            "</output_contract>",
        ]
    )


def profile_skill_path() -> Path:
    """Return the packaged Codex profile skill path."""

    return package_resource_path("resources", "codex", "skills", "tonepath-profile", "SKILL.md")


def profile_schema_path() -> Path:
    """Return the packaged Codex profile suggestion schema path."""

    return package_resource_path("resources", "codex", "skills", "tonepath-profile", "schemas", "profile-suggestions.schema.json")


def package_resource_path(*parts: str) -> Path:
    """Return a packaged Tonepath resource path."""

    return Path(str(resources.files("tonepath").joinpath(*parts)))


def memory_context_text(paths: list[Path]) -> str:
    """Return Markdown memory context from existing local paths."""

    parts: list[str] = []
    for path in paths:
        if path.exists():
            parts.append(f"<!-- source: {path.name} -->\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def markdown_document(title: str, generated: str, human_notes: str) -> str:
    """Return a Markdown document with generated and human-editable sections."""

    notes = human_notes.strip() or DEFAULT_HUMAN_NOTES
    return "\n".join(
        [
            f"# {title}",
            "",
            GENERATED_START,
            generated.rstrip(),
            GENERATED_END,
            "",
            "## Human Notes",
            "",
            HUMAN_NOTES_START,
            notes,
            HUMAN_NOTES_END,
            "",
        ]
    )


def extract_human_notes(text: str) -> str:
    """Extract preserved human notes from a generated Markdown document."""

    start_marker = text.find(HUMAN_NOTES_START)
    if start_marker == -1:
        return DEFAULT_HUMAN_NOTES
    start = start_marker + len(HUMAN_NOTES_START)
    end = text.find(HUMAN_NOTES_END, start)
    if end == -1:
        return DEFAULT_HUMAN_NOTES
    return text[start:end].strip() or DEFAULT_HUMAN_NOTES


def markdown_kv_table(value: object) -> str:
    """Render a simple key/value object as a Markdown table."""

    if not isinstance(value, dict) or not value:
        return "_No data yet._"
    rows = ["| Key | Value |", "|---|---|"]
    for key in sorted(value):
        rows.append(f"| {markdown_cell(key)} | {markdown_cell(value[key])} |")
    return "\n".join(rows)


def markdown_rule_table(rules: list[ProfileRule]) -> str:
    """Render active profile rules as a Markdown table."""

    if not rules:
        return "_No active profile rules._"
    rows = ["| Rule | Scope | Source | Confidence | Rationale |", "|---|---|---|---|---|"]
    for rule in rules:
        payload = json.loads(rule.value)
        rows.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(payload.get("rule_type")),
                    markdown_cell(payload.get("scope")),
                    markdown_cell(rule.source),
                    markdown_cell(rule.confidence),
                    markdown_cell(payload.get("rationale")),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def markdown_suggestion_table(suggestions: list[dict[str, object]]) -> str:
    """Render pending profile suggestions as a Markdown table."""

    if not suggestions:
        return "_No pending profile suggestions._"
    rows = ["| Suggestion | Scope | Rule | Source | Confidence | Rationale |", "|---|---|---|---|---|---|"]
    for item in suggestions[:20]:
        rows.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(item.get("suggestion_id")),
                    markdown_cell(item.get("scope")),
                    markdown_cell(item.get("rule_type")),
                    markdown_cell(item.get("source")),
                    markdown_cell(item.get("confidence")),
                    markdown_cell(item.get("rationale")),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def markdown_feedback_table(evidence: dict[str, object]) -> str:
    """Render profile evidence feedback events as a Markdown table."""

    events = [event for event in evidence.get("feedback_events", []) if isinstance(event, dict)]
    if not events:
        return "_Not enough feedback yet._"
    rows = ["| Feedback | Prompt | Track | BPM | Loudness | Energy | Vocalness | Source |", "|---|---|---|---:|---:|---:|---:|---|"]
    for event in events[-20:]:
        track = event.get("track") if isinstance(event.get("track"), dict) else {}
        features = track.get("features") if isinstance(track, dict) and isinstance(track.get("features"), dict) else {}
        rows.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(event.get("feedback_type")),
                    markdown_cell(event.get("prompt")),
                    markdown_cell(track.get("label") if isinstance(track, dict) else None),
                    markdown_number(features.get("bpm") if isinstance(features, dict) else None),
                    markdown_number(features.get("loudness") if isinstance(features, dict) else None),
                    markdown_number(features.get("energy") if isinstance(features, dict) else None),
                    markdown_number(features.get("vocalness") if isinstance(features, dict) else None),
                    markdown_cell(features.get("source") if isinstance(features, dict) else None),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def markdown_reaction_table(evidence: dict[str, object]) -> str:
    """Render current stable track reactions as a Markdown table."""

    reactions = [reaction for reaction in evidence.get("track_reactions", []) if isinstance(reaction, dict)]
    if not reactions:
        return "_No track reactions yet._"
    rows = ["| Reaction | Track | Vocalness | Updated |", "|---|---|---:|---|"]
    for reaction in reactions[:20]:
        rows.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(reaction.get("reaction")),
                    markdown_cell(reaction.get("label")),
                    markdown_number(reaction_feature_value(reaction, "vocalness")),
                    markdown_cell(reaction.get("updated_at")),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def markdown_cell(value: object) -> str:
    """Return a safe Markdown table cell."""

    if value is None:
        return "--"
    return str(value).replace("\n", " ").replace("|", "\\|")


def markdown_number(value: object) -> str:
    """Return a compact Markdown number value."""

    if value is None:
        return "--"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return markdown_cell(value)


def feature_value(event: dict[str, object], field: str) -> object | None:
    """Return a feature value from one evidence event."""

    track = event.get("track")
    if not isinstance(track, dict):
        return None
    features = track.get("features")
    if not isinstance(features, dict):
        return None
    return features.get(field)


def reaction_feature_value(reaction: dict[str, object], field: str) -> object | None:
    """Return a feature value from one stable reaction record."""

    features = reaction.get("features")
    if not isinstance(features, dict):
        return None
    return features.get(field)


def default_target(rule_type: str) -> str:
    """Return the target feature for a rule type."""

    if rule_type == "prefer_artist":
        return "artist"
    if rule_type == "demote_high_bpm":
        return "bpm"
    return rule_type.removeprefix("prefer_lower_")


def default_threshold(rule_type: str) -> float:
    """Return a conservative default threshold for a rule type."""

    return {"prefer_lower_loudness": -12.0, "prefer_lower_energy": 0.45, "prefer_lower_vocalness": 0.35, "demote_high_bpm": 135.0}.get(
        rule_type, 0.0
    )


def float_or_default(value: object, default: float) -> float:
    """Return a float value or a default."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_or_default(value: object, default: int) -> int:
    """Return an integer value or a default."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def display_title_from_values(title: object, album: object) -> str:
    """Return a display-safe title without exposing file paths."""

    return display_title(fake_track(title, None, album))


def display_artist_from_values(artist: object) -> str:
    """Return a display-safe artist without exposing file paths."""

    return display_artist(fake_track(None, artist, None))


def display_label_from_values(title: object, artist: object, album: object) -> str:
    """Return a display-safe label without exposing file paths."""

    return display_label(fake_track(title, artist, album))


def fake_track(title: object, artist: object, album: object) -> Any:
    """Return the minimal track-like object needed by display helpers."""

    class TrackLike:
        path = Path("track")
        duration = None
        format = None
        file_hash = ""
        mtime = 0.0
        genre = None
        id = None

    item = TrackLike()
    item.title = title if isinstance(title, str) else None
    item.artist = artist if isinstance(artist, str) else None
    item.album = album if isinstance(album, str) else None
    return item
