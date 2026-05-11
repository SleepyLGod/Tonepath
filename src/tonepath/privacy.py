"""Privacy-facing operations for Tonepath."""

from __future__ import annotations

from tonepath import config
from tonepath.db import TonepathStore


def privacy_status(store: TonepathStore) -> str:
    """Return a human-readable local privacy summary."""

    summary = store.profile_summary()
    lines = [
        "Tonepath privacy status",
        f"Data location: {config.db_path()}",
        "Network: offline by default",
        "LLM: disabled by default",
        "Sent to LLM: none",
        "Stored locally:",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def delete_profile(store: TonepathStore) -> None:
    """Delete local profile/session/feedback data."""

    store.delete_profile_data()

