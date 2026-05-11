"""Environment checks for Tonepath."""

from __future__ import annotations

from pathlib import Path

from tonepath import config
from tonepath.playback import MpvAdapter


def run_doctor() -> str:
    """Return a report about local Tonepath dependencies."""

    settings = config.load_config()
    data_dir = config.ensure_data_dir()
    music_dir_lines = music_directory_report(settings.expanded_music_dirs())
    player_status = player_report(settings.player)
    lines = [
        "Tonepath doctor",
        f"Config path: {config.config_path()}",
        f"Data directory: {data_dir}",
        f"SQLite path: {config.db_path()}",
        f"Player: {settings.player}",
        f"Network mode: {settings.network_mode}",
        f"Privacy send_to_llm: {settings.privacy.send_to_llm}",
        f"Privacy store_play_history: {settings.privacy.store_play_history}",
        "Music directories:",
        *music_dir_lines,
        player_status,
    ]
    if settings.player == "mpv" and "missing" in player_status:
        lines.append("Install mpv to enable playback: brew install mpv")
    return "\n".join(lines)


def music_directory_report(paths: tuple[Path, ...]) -> list[str]:
    """Return validation lines for configured music directories."""

    if not paths:
        return ["  none configured"]
    lines: list[str] = []
    for path in paths:
        status = "ok" if path.is_dir() else "missing"
        lines.append(f"  {path}: {status}")
    return lines


def player_report(player: str) -> str:
    """Return a player availability line."""

    if player != "mpv":
        return f"{player}: configured, but only mpv is implemented in v0"
    mpv = MpvAdapter()
    return f"mpv: {'ok' if mpv.available() else 'missing'}"
