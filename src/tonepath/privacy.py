"""Local privacy inventory, export, and deletion controls."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.display import METADATA_ARTIST_FIELD, METADATA_OVERRIDE_SOURCE, METADATA_TITLE_FIELD, clean_metadata_text
from tonepath.llm import provider_config
from tonepath.memory import (
    LAST_CONSOLIDATED_SEQUENCE_KEY,
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    read_memory_logs,
    redact_memory_records,
    redact_memory_text,
)


PRIVACY_CATEGORY_IDS = (
    "memory",
    "personalization",
    "history",
    "library-evidence",
    "models-storage",
)
DELETABLE_CATEGORY_IDS = ("memory", "personalization", "history")
ALL_PERSONAL_CATEGORIES = DELETABLE_CATEGORY_IDS

_CATEGORY_METADATA: dict[str, dict[str, object]] = {
    "memory": {
        "label": "Memory",
        "description": "Private notes, the editable Memory Profile, and consolidation evidence.",
        "sensitivity": "high",
        "rebuildable": False,
        "capabilities": ("inspect", "export", "delete"),
        "effects": ("Deleting Memory removes notes and the consolidation checkpoint.",),
    },
    "personalization": {
        "label": "Personalization",
        "description": "Track reactions, feedback, active preference rules, profile evidence, and pending suggestions.",
        "sensitivity": "high",
        "rebuildable": False,
        "capabilities": ("inspect", "export", "delete"),
        "effects": ("Deleting Personalization removes future recommendation adjustments.",),
    },
    "history": {
        "label": "Listening History",
        "description": "Requests, paths, queue snapshots, bookmarks, plays, and audit evidence.",
        "sensitivity": "high",
        "rebuildable": False,
        "capabilities": ("inspect", "export", "delete"),
        "effects": ("Deleting History removes saved and replayable listening paths.",),
    },
    "library-evidence": {
        "label": "Library Evidence",
        "description": "Scanned tracks, audio features, enrichment, embeddings, and separated evidence.",
        "sensitivity": "medium",
        "rebuildable": True,
        "capabilities": ("inspect",),
        "effects": ("Read-only in Privacy Center v1; original music is never included.",),
    },
    "models-storage": {
        "label": "Models & Storage",
        "description": "Local model files, isolated runtimes, and package caches.",
        "sensitivity": "low",
        "rebuildable": True,
        "capabilities": ("inspect",),
        "effects": ("Read-only in Privacy Center v1.",),
    },
}

_CATEGORY_TABLES: dict[str, tuple[str, ...]] = {
    "memory": (),
    "personalization": ("track_reactions", "feedback", "profile_rules"),
    "history": ("sessions", "session_phases", "session_queue_items", "session_bookmarks", "plays"),
    "library-evidence": ("tracks", "track_features", "track_enrichment"),
    "models-storage": (),
}


@dataclass(frozen=True)
class PrivacyCategoryReport:
    """Inventory report for one stable local data category."""

    id: str
    label: str
    description: str
    sensitivity: str
    records: dict[str, int]
    file_count: int
    file_size_bytes: int
    locations: tuple[str, ...]
    rebuildable: bool
    capabilities: tuple[str, ...]
    effects: tuple[str, ...]

    @property
    def record_count(self) -> int:
        """Return the total database rows represented by this category."""

        return sum(self.records.values())

    def to_payload(self) -> dict[str, object]:
        """Return a stable JSON-ready category payload."""

        payload = asdict(self)
        payload["record_count"] = self.record_count
        return payload


@dataclass(frozen=True)
class PrivacyInventory:
    """Read-only snapshot of Tonepath's local data footprint."""

    schema: str
    generated_at: str
    data_home: str
    home_exists: bool
    database_exists: bool
    shared_database_size_bytes: int
    categories: tuple[PrivacyCategoryReport, ...]
    external_processing: dict[str, object]
    protected: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return a stable JSON-ready inventory payload."""

        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "data_home": self.data_home,
            "home_exists": self.home_exists,
            "database_exists": self.database_exists,
            "shared_database_size_bytes": self.shared_database_size_bytes,
            "categories": [category.to_payload() for category in self.categories],
            "external_processing": self.external_processing,
            "protected": list(self.protected),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PrivacyDeletePlan:
    """Immutable preview of one personal-data deletion operation."""

    schema: str
    categories: tuple[str, ...]
    will_delete: tuple[str, ...]
    retained: tuple[str, ...]
    warnings: tuple[str, ...]
    fingerprint: str

    def to_payload(self) -> dict[str, object]:
        """Return a stable JSON-ready deletion preview."""

        return {
            "schema": self.schema,
            "categories": list(self.categories),
            "will_delete": list(self.will_delete),
            "retained": list(self.retained),
            "warnings": list(self.warnings),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PrivacyDeleteItemResult:
    """Result for one database or filesystem deletion component."""

    category: str
    component: str
    status: str
    message: str


@dataclass(frozen=True)
class PrivacyDeleteResult:
    """Detailed result of an executed privacy deletion plan."""

    schema: str
    plan_fingerprint: str
    items: tuple[PrivacyDeleteItemResult, ...]

    @property
    def failed(self) -> bool:
        """Return whether any deletion component failed."""

        return any(item.status == "failed" for item in self.items)

    @property
    def changed_categories(self) -> tuple[str, ...]:
        """Return categories for which at least one component was deleted."""

        return tuple(
            category
            for category in DELETABLE_CATEGORY_IDS
            if any(item.category == category and item.status == "deleted" for item in self.items)
        )

    def to_payload(self) -> dict[str, object]:
        """Return a stable JSON-ready deletion result."""

        return {
            "schema": self.schema,
            "plan_fingerprint": self.plan_fingerprint,
            "failed": self.failed,
            "changed_categories": list(self.changed_categories),
            "items": [asdict(item) for item in self.items],
        }


def build_privacy_inventory(store: TonepathStore | None = None) -> PrivacyInventory:
    """Inspect known Tonepath data without creating storage or following symlinks."""

    home = config.data_dir()
    database = config.db_path()
    connection, close_connection, database_warning = _inventory_connection(database, store)
    try:
        table_counts = _table_counts(connection)
    finally:
        if close_connection and connection is not None:
            connection.close()

    categories: list[PrivacyCategoryReport] = []
    scan_warnings: list[str] = []
    for category_id in PRIVACY_CATEGORY_IDS:
        roots = _category_roots(home, category_id)
        file_count = 0
        file_size = 0
        for root in roots:
            count, size, warnings = _safe_tree_stats(root)
            file_count += count
            file_size += size
            scan_warnings.extend(warnings)
        metadata = _CATEGORY_METADATA[category_id]
        categories.append(
            PrivacyCategoryReport(
                id=category_id,
                label=str(metadata["label"]),
                description=str(metadata["description"]),
                sensitivity=str(metadata["sensitivity"]),
                records={table: table_counts.get(table, 0) for table in _CATEGORY_TABLES[category_id]},
                file_count=file_count,
                file_size_bytes=file_size,
                locations=tuple(str(root) for root in roots),
                rebuildable=bool(metadata["rebuildable"]),
                capabilities=tuple(str(value) for value in metadata["capabilities"]),
                effects=tuple(str(value) for value in metadata["effects"]),
            )
        )

    settings = config.load_config()
    try:
        provider = provider_config()
        provider_name = provider.provider
        key_present = provider.configured
    except ValueError:
        provider_name = "unsupported"
        key_present = False
    warnings = [*scan_warnings]
    if database_warning:
        warnings.append(database_warning)
    return PrivacyInventory(
        schema="tonepath-privacy-inventory-v1",
        generated_at=_utc_now(),
        data_home=str(home),
        home_exists=home.exists(),
        database_exists=database.is_file() and not database.is_symlink(),
        shared_database_size_bytes=_regular_file_size(database),
        categories=tuple(categories),
        external_processing={
            "allowed": settings.privacy.send_to_llm,
            "network_mode": settings.network_mode,
            "provider": provider_name,
            "key_present": key_present,
            "transmission_history": "not recorded",
        },
        protected=(
            "Tonepath configuration",
            "original music files and configured music directories",
            "library evidence and audio features",
            "models, runtimes, and package caches",
        ),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def privacy_status(store: TonepathStore | None = None) -> str:
    """Return a compact human-readable local privacy summary."""

    return privacy_status_from_inventory(build_privacy_inventory(store))


def render_privacy_inventory(inventory: PrivacyInventory) -> str:
    """Render a detailed inventory for CLI inspection."""

    lines = [privacy_status_from_inventory(inventory), "", "Category details:"]
    for category in inventory.categories:
        capability = ", ".join(category.capabilities)
        lines.extend(
            [
                f"\n{category.label} ({category.id}) · sensitivity {category.sensitivity}",
                category.description,
                f"Records: {category.record_count} · Files: {category.file_count} · Size: {_human_bytes(category.file_size_bytes)}",
                f"Capabilities: {capability}",
                *(f"Location: {location}" for location in category.locations),
                *(f"Note: {effect}" for effect in category.effects),
            ]
        )
    if inventory.warnings:
        lines.extend(("", "Warnings:", *(f"- {warning}" for warning in inventory.warnings)))
    return "\n".join(lines)


def privacy_status_from_inventory(inventory: PrivacyInventory) -> str:
    """Render the compact status section from an existing inventory."""

    external = inventory.external_processing
    lines = [
        "Tonepath Data & Privacy",
        f"Data home: {inventory.data_home}",
        f"Database: {'present' if inventory.database_exists else 'not created'}",
        (
            "External processing: "
            f"{'allowed' if external['allowed'] else 'off'} · provider {external['provider']} · "
            f"key {'present' if external['key_present'] else 'missing'}"
        ),
        "Transmission history: not recorded",
        "Local data:",
    ]
    for category in inventory.categories:
        lines.append(
            f"- {category.label}: {category.record_count} records, "
            f"{category.file_count} {_plural(category.file_count, 'file')}, {_human_bytes(category.file_size_bytes)}"
        )
    return "\n".join(lines)


def export_personal_data(output: Path, store: TonepathStore | None = None) -> Path:
    """Export a sanitized, owner-only copy of personal Tonepath data."""

    output = output.expanduser()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise RuntimeError(f"Export output is not a normal directory: {output}")
        if any(output.iterdir()):
            raise RuntimeError(f"Export output directory is not empty: {output}")
    else:
        output.mkdir(parents=True, mode=PRIVATE_DIR_MODE)
    output.chmod(PRIVATE_DIR_MODE)
    memory_output = output / "memory"
    memory_output.mkdir(mode=PRIVATE_DIR_MODE)
    memory_output.chmod(PRIVATE_DIR_MODE)

    home = config.data_dir()
    profile_text = ""
    profile_path = home / "memory" / "profile.md"
    if _safe_regular_file(profile_path):
        profile_text = profile_path.read_text(encoding="utf-8")
    redacted_profile, _ = redact_memory_text(profile_text)
    _write_private_text(memory_output / "profile.md", redacted_profile)

    logs = read_memory_logs() if _safe_regular_file(home / "memory" / "logs" / "memory-log.jsonl") else []
    redacted_logs, _ = redact_memory_records(logs)
    log_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in redacted_logs)
    _write_private_text(memory_output / "memory-log.jsonl", log_text)

    connection, close_connection, database_warning = _inventory_connection(config.db_path(), store)
    try:
        personalization = _export_personalization(connection)
        history = _export_history(connection)
    finally:
        if close_connection and connection is not None:
            connection.close()
    if database_warning:
        personalization["warnings"] = [database_warning]
        history["warnings"] = [database_warning]

    settings = config.load_config()
    try:
        provider = provider_config()
        provider_name = provider.provider
        key_present = provider.configured
    except ValueError:
        provider_name = "unsupported"
        key_present = False
    settings_payload = {
        "experience_mode": settings.experience.mode,
        "network_mode": settings.network_mode,
        "player": settings.player,
        "privacy": {
            "send_to_llm": settings.privacy.send_to_llm,
            "store_play_history": settings.privacy.store_play_history,
        },
        "models": {
            "mode": settings.models.mode,
            "allow_setup": settings.models.allow_setup,
            "allow_online": settings.models.allow_online,
            "preferred_tagger": settings.models.preferred_tagger,
        },
        "ui": {"theme": settings.ui.theme},
        "external_processing": {
            "provider": provider_name,
            "key_present": key_present,
            "transmission_history": "not recorded",
        },
    }

    _write_private_json(output / "personalization.json", _sanitize_payload(personalization))
    _write_private_json(output / "history.json", _sanitize_payload(history))
    _write_private_json(output / "settings.json", settings_payload)
    _write_private_text(
        output / "README.md",
        "# Tonepath Personal Data Export\n\n"
        "This owner-only bundle contains Memory, personalization, and listening history.\n"
        "It excludes API keys, absolute music paths, audio files, the SQLite database, models, runtimes, and rebuildable caches.\n"
        "Tonepath does not record a history of optional LLM transmissions.\n",
    )
    manifest = {
        "schema": "tonepath-personal-data-export-v1",
        "exported_at": _utc_now(),
        "files": [
            "README.md",
            "memory/profile.md",
            "memory/memory-log.jsonl",
            "personalization.json",
            "history.json",
            "settings.json",
        ],
        "privacy": {
            "owner_only": True,
            "contains_api_keys": False,
            "contains_absolute_music_paths": False,
            "contains_audio": False,
            "contains_database": False,
        },
    }
    _write_private_json(output / "manifest.json", manifest)
    return output


def plan_privacy_delete(
    categories: tuple[str, ...] | list[str],
    store: TonepathStore | None = None,
) -> PrivacyDeletePlan:
    """Build a zero-write deletion preview for the requested personal categories."""

    normalized = _normalize_delete_categories(categories)
    inventory = build_privacy_inventory(store)
    report_by_id = {report.id: report for report in inventory.categories}
    will_delete: list[str] = []
    warnings = [
        "Deletion removes data from Tonepath active storage, not from system backups, APFS snapshots, or SSD forensic recovery.",
    ]
    for category in normalized:
        report = report_by_id[category]
        will_delete.append(
            f"{report.label}: {report.record_count} database records and "
            f"{report.file_count} {_plural(report.file_count, 'file')} ({_human_bytes(report.file_size_bytes)})"
        )
    if "memory" in normalized and "personalization" not in normalized:
        active_rules = report_by_id["personalization"].records.get("profile_rules", 0)
        if active_rules:
            warnings.append(
                f"{active_rules} active rules will remain and may continue to affect future recommendations."
            )
    retained = [
        "Tonepath configuration",
        "original music files and configured music directories",
        "library evidence, tracks, and audio features",
        "models, runtimes, and package caches",
    ]
    if "memory" not in normalized:
        retained.append("Memory notes and Memory Profile")
    if "personalization" not in normalized:
        retained.append("feedback and active preference rules")
    if "history" not in normalized:
        retained.append("listening history and saved sessions")
    return PrivacyDeletePlan(
        schema="tonepath-privacy-delete-plan-v1",
        categories=normalized,
        will_delete=tuple(will_delete),
        retained=tuple(retained),
        warnings=tuple(warnings),
        fingerprint=_inventory_fingerprint(inventory, normalized),
    )


def execute_privacy_delete(
    plan: PrivacyDeletePlan,
    store: TonepathStore | None = None,
) -> PrivacyDeleteResult:
    """Execute a current deletion plan and report each component independently."""

    current = plan_privacy_delete(plan.categories, store=store)
    if current.fingerprint != plan.fingerprint:
        raise RuntimeError("Privacy data changed since the preview; refresh the plan and confirm again.")

    owned_store = False
    mutable_store = store
    if mutable_store is None and _safe_regular_file(config.db_path()):
        mutable_store = TonepathStore(config.db_path())
        owned_store = True
    results: list[PrivacyDeleteItemResult] = []
    try:
        if mutable_store is not None:
            mutable_store.conn.execute("PRAGMA secure_delete = ON")
        for category in plan.categories:
            results.append(_delete_database_category(category, mutable_store))
            for root in _deletion_roots(config.data_dir(), category):
                results.append(_delete_path_component(category, root))
    finally:
        if owned_store and mutable_store is not None:
            mutable_store.close()
    return PrivacyDeleteResult(
        schema="tonepath-privacy-delete-result-v1",
        plan_fingerprint=plan.fingerprint,
        items=tuple(results),
    )


def delete_profile(store: TonepathStore) -> None:
    """Delete local personalization and history while preserving library evidence."""

    store.conn.execute("PRAGMA secure_delete = ON")
    store.delete_profile_data()


def render_delete_plan(plan: PrivacyDeletePlan) -> str:
    """Render a human-readable deletion preview."""

    lines = ["Privacy deletion preview", f"Categories: {', '.join(plan.categories)}", "Will delete:"]
    lines.extend(f"- {item}" for item in plan.will_delete)
    lines.append("Will keep:")
    lines.extend(f"- {item}" for item in plan.retained)
    lines.append("Warnings:")
    lines.extend(f"- {warning}" for warning in plan.warnings)
    return "\n".join(lines)


def render_delete_result(result: PrivacyDeleteResult) -> str:
    """Render a component-by-component deletion result."""

    lines = ["Privacy deletion result"]
    lines.extend(f"- {item.status.upper()} · {item.component}: {item.message}" for item in result.items)
    if result.failed:
        lines.append("Some components failed. Review the items above and rerun the same delete command.")
    return "\n".join(lines)


def _category_roots(home: Path, category: str) -> tuple[Path, ...]:
    if category == "memory":
        return (home / "memory", home / "cache" / "memory")
    if category == "personalization":
        return (home / "profile", home / "cache" / "profile")
    if category == "history":
        return (home / "cache" / "audit",)
    if category == "library-evidence":
        return (home / "cache" / "embeddings", home / "cache" / "separated")
    if category == "models-storage":
        return (home / "cache" / "models", home / "cache" / "pip", home / "cache" / "uv", home / "runtimes")
    raise ValueError(f"Unknown privacy category: {category}")


def _deletion_roots(home: Path, category: str) -> tuple[Path, ...]:
    if category not in DELETABLE_CATEGORY_IDS:
        raise ValueError(f"Privacy category is read-only: {category}")
    return _category_roots(home, category)


def _inventory_connection(
    database: Path,
    store: TonepathStore | None,
) -> tuple[sqlite3.Connection | None, bool, str | None]:
    if store is not None:
        return store.conn, False, None
    if not _safe_regular_file(database):
        return None, False, None
    uri = f"file:{quote(str(database))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection, True, None
    except sqlite3.Error as exc:
        return None, False, f"Database could not be inspected: {exc}"


def _table_counts(connection: sqlite3.Connection | None) -> dict[str, int]:
    if connection is None:
        return {}
    tables = {
        str(row["name"])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    counts: dict[str, int] = {}
    for table in {table for values in _CATEGORY_TABLES.values() for table in values}:
        if table not in tables:
            counts[table] = 0
            continue
        row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        counts[table] = int(row["count"]) if row is not None else 0
    return counts


def _safe_tree_stats(root: Path) -> tuple[int, int, list[str]]:
    if root.is_symlink() or not root.exists():
        return 0, 0, []
    if root.is_file():
        return 1, _regular_file_size(root), []
    file_count = 0
    total_size = 0
    warnings: list[str] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            file_count += 1
                            total_size += entry.stat(follow_symlinks=False).st_size
                    except OSError as exc:
                        warnings.append(f"Could not inspect {entry.path}: {exc}")
        except OSError as exc:
            warnings.append(f"Could not inspect {directory}: {exc}")
    return file_count, total_size, warnings


def _inventory_fingerprint(inventory: PrivacyInventory, categories: tuple[str, ...]) -> str:
    reports = {report.id: report for report in inventory.categories}
    payload: dict[str, object] = {
        "categories": [reports[category].to_payload() for category in categories],
        "database": _path_fingerprint(config.db_path()),
        "paths": [],
    }
    paths: list[dict[str, object]] = []
    for category in categories:
        for root in _deletion_roots(config.data_dir(), category):
            paths.extend(_tree_fingerprint(root))
    payload["paths"] = paths
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_fingerprint(root: Path) -> list[dict[str, object]]:
    if root.is_symlink():
        stat_result = root.lstat()
        return [{"root": str(root), "kind": "symlink", "mtime_ns": stat_result.st_mtime_ns}]
    if not root.exists():
        return [{"root": str(root), "kind": "absent"}]
    records: list[dict[str, object]] = []
    if root.is_file():
        return [_path_fingerprint(root)]
    pending = [root]
    while pending:
        directory = pending.pop()
        records.append(_path_fingerprint(directory))
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            records.append({"path": str(directory), "error": str(exc)})
            continue
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                stat_result = entry.stat(follow_symlinks=False)
                records.append({"path": str(path), "kind": "symlink", "mtime_ns": stat_result.st_mtime_ns})
            elif entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                records.append(_path_fingerprint(path))
    return sorted(records, key=lambda record: str(record.get("path") or record.get("root")))


def _path_fingerprint(path: Path) -> dict[str, object]:
    if path.is_symlink():
        stat_result = path.lstat()
        return {"path": str(path), "kind": "symlink", "mtime_ns": stat_result.st_mtime_ns}
    if not path.exists():
        return {"path": str(path), "kind": "absent"}
    stat_result = path.stat()
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _normalize_delete_categories(categories: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    requested = set(categories)
    if not requested:
        raise ValueError("Choose at least one privacy category to delete.")
    unknown = requested.difference(PRIVACY_CATEGORY_IDS)
    if unknown:
        raise ValueError(f"Unknown privacy category: {', '.join(sorted(unknown))}")
    readonly = requested.difference(DELETABLE_CATEGORY_IDS)
    if readonly:
        raise ValueError(f"Privacy category is read-only in v1: {', '.join(sorted(readonly))}")
    return tuple(category for category in DELETABLE_CATEGORY_IDS if category in requested)


def _delete_database_category(
    category: str,
    store: TonepathStore | None,
) -> PrivacyDeleteItemResult:
    component = f"SQLite {category} records"
    if store is None:
        return PrivacyDeleteItemResult(category, component, "already_absent", "Database is not present.")
    before = store.profile_summary()
    try:
        if category == "memory":
            existed = store.get_app_state(LAST_CONSOLIDATED_SEQUENCE_KEY) is not None
            store.delete_app_state(LAST_CONSOLIDATED_SEQUENCE_KEY)
            deleted = 1 if existed else 0
        elif category == "personalization":
            deleted = before["track_reactions"] + before["feedback"] + before["profile_rules"]
            store.delete_personalization_data()
        elif category == "history":
            deleted = sum(before[table] for table in ("sessions", "session_phases", "session_queue_items", "session_bookmarks", "plays"))
            store.delete_history_data(preserve_feedback=True)
        else:
            raise ValueError(f"Privacy category is read-only: {category}")
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        return PrivacyDeleteItemResult(category, component, "failed", str(exc))
    status = "deleted" if deleted else "already_absent"
    message = f"Removed {deleted} records." if deleted else "No matching records were present."
    return PrivacyDeleteItemResult(category, component, status, message)


def _delete_path_component(category: str, path: Path) -> PrivacyDeleteItemResult:
    component = str(path)
    if not path.exists() and not path.is_symlink():
        return PrivacyDeleteItemResult(category, component, "already_absent", "Path is not present.")
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        return PrivacyDeleteItemResult(category, component, "failed", str(exc))
    return PrivacyDeleteItemResult(category, component, "deleted", "Removed from Tonepath active storage.")


def _export_personalization(connection: sqlite3.Connection | None) -> dict[str, object]:
    return {
        "schema": "tonepath-personalization-export-v1",
        "track_reactions": _export_track_reactions(connection),
        "feedback": _select_rows(
            connection,
            "feedback",
            "id, session_id, track_id, type, value, created_at",
            "ORDER BY created_at, id",
        ),
        "profile_rules": _select_rows(
            connection,
            "profile_rules",
            "id, key, value, source, confidence, created_at, updated_at",
            "ORDER BY created_at, id",
        ),
    }


def _export_track_reactions(connection: sqlite3.Connection | None) -> list[dict[str, object]]:
    """Return current reactions with display metadata and no local paths."""

    if connection is None or not _table_exists(connection, "track_reactions"):
        return []
    rows = connection.execute(
        """
        SELECT
          track_reactions.track_id,
          track_reactions.reaction,
          tracks.title,
          tracks.artist,
          (
            SELECT value
            FROM track_enrichment
            WHERE track_id = tracks.id AND field = ? AND source = ?
            ORDER BY id DESC
            LIMIT 1
          ) AS title_override,
          (
            SELECT value
            FROM track_enrichment
            WHERE track_id = tracks.id AND field = ? AND source = ?
            ORDER BY id DESC
            LIMIT 1
          ) AS artist_override
        FROM track_reactions
        JOIN tracks ON tracks.id = track_reactions.track_id
        ORDER BY track_reactions.updated_at, track_reactions.track_id
        """,
        (
            METADATA_TITLE_FIELD,
            METADATA_OVERRIDE_SOURCE,
            METADATA_ARTIST_FIELD,
            METADATA_OVERRIDE_SOURCE,
        ),
    ).fetchall()
    return [
        {
            "track_id": int(row["track_id"]),
            "reaction": str(row["reaction"]),
            "title": clean_metadata_text(row["title_override"]) or clean_metadata_text(row["title"]) or "unknown",
            "artist": clean_metadata_text(row["artist_override"]) or clean_metadata_text(row["artist"]) or "unknown",
        }
        for row in rows
    ]


def _export_history(connection: sqlite3.Connection | None) -> dict[str, object]:
    queue = _select_rows(
        connection,
        "session_queue_items",
        "session_id, position, title, artist, phase_label, score, confidence, reasons_json",
        "ORDER BY session_id, position",
    )
    for item in queue:
        raw_reasons = item.pop("reasons_json", "[]")
        try:
            reasons = json.loads(str(raw_reasons))
        except json.JSONDecodeError:
            reasons = []
        item["reasons"] = reasons if isinstance(reasons, list) else []
    return {
        "schema": "tonepath-history-export-v1",
        "sessions": _select_rows(connection, "sessions", "*", "ORDER BY started_at, id"),
        "phases": _select_rows(connection, "session_phases", "*", "ORDER BY session_id, start_sec, id"),
        "queue": queue,
        "bookmarks": _select_rows(connection, "session_bookmarks", "*", "ORDER BY saved_at, session_id"),
        "plays": _select_rows(
            connection,
            "plays",
            "id, session_id, track_id, phase_id, started_at, ended_at, skipped, position_sec",
            "ORDER BY started_at, id",
        ),
    }


def _select_rows(
    connection: sqlite3.Connection | None,
    table: str,
    columns: str,
    suffix: str,
) -> list[dict[str, object]]:
    if connection is None:
        return []
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if exists is None:
        return []
    return [dict(row) for row in connection.execute(f"SELECT {columns} FROM {table} {suffix}").fetchall()]


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    """Return whether one SQLite table exists."""

    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_memory_text(value)[0]
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item) for item in value]
    return value


def _write_private_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(PRIVATE_FILE_MODE)


def _write_private_json(path: Path, payload: object) -> None:
    _write_private_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _safe_regular_file(path: Path) -> bool:
    return path.exists() and path.is_file() and not path.is_symlink()


def _regular_file_size(path: Path) -> int:
    if not _safe_regular_file(path):
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _plural(value: int, singular: str) -> str:
    return singular if value == 1 else f"{singular}s"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
