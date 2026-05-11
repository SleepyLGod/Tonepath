"""Configuration helpers for local Tonepath state."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


APP_DIR_NAME = ".tonepath"
CONFIG_FILENAME = "config.toml"
DB_FILENAME = "tonepath.db"


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy defaults from the local config file."""

    send_to_llm: bool = False
    store_play_history: bool = True


@dataclass(frozen=True)
class TonepathConfig:
    """User-editable local Tonepath configuration."""

    music_dirs: tuple[str, ...]
    data_dir: str
    player: str
    network_mode: str
    privacy: PrivacyConfig

    def expanded_music_dirs(self) -> tuple[Path, ...]:
        """Return configured music directories with `~` expanded."""

        return tuple(Path(path).expanduser() for path in self.music_dirs)

    def expanded_data_dir(self) -> Path:
        """Return the configured data directory with `~` expanded."""

        return Path(self.data_dir).expanduser()


def app_home() -> Path:
    """Return the Tonepath app home directory."""

    override = os.environ.get("TONEPATH_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / APP_DIR_NAME


def config_path() -> Path:
    """Return the local TOML config path."""

    return app_home() / CONFIG_FILENAME


def default_config() -> TonepathConfig:
    """Return the default local-first Tonepath configuration."""

    default_root = os.environ.get("TONEPATH_HOME", f"~/{APP_DIR_NAME}")
    return TonepathConfig(
        music_dirs=("~/Music",),
        data_dir=default_root,
        player="mpv",
        network_mode="offline",
        privacy=PrivacyConfig(),
    )


def load_config() -> TonepathConfig:
    """Load the local TOML config, returning defaults when it does not exist."""

    path = config_path()
    defaults = default_config()
    if not path.exists():
        return defaults

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    privacy_data = data.get("privacy", {})
    data_dir_override = os.environ.get("TONEPATH_HOME")
    return TonepathConfig(
        music_dirs=tuple(str(item) for item in data.get("music_dirs", defaults.music_dirs)),
        data_dir=str(data_dir_override or data.get("data_dir", defaults.data_dir)),
        player=str(data.get("player", defaults.player)),
        network_mode=str(data.get("network_mode", defaults.network_mode)),
        privacy=PrivacyConfig(
            send_to_llm=bool(privacy_data.get("send_to_llm", defaults.privacy.send_to_llm)),
            store_play_history=bool(privacy_data.get("store_play_history", defaults.privacy.store_play_history)),
        ),
    )


def write_config(config: TonepathConfig, overwrite: bool = True) -> Path:
    """Write the local TOML config and return its path."""

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    path.write_text(render_config(config), encoding="utf-8")
    return path


def init_config(overwrite: bool = False) -> Path:
    """Create a default config file if one does not already exist."""

    return write_config(default_config(), overwrite=overwrite)


def add_music_dir(path: Path) -> TonepathConfig:
    """Persist a new music directory if it is not already configured."""

    current = load_config()
    value = str(path.expanduser())
    music_dirs = list(current.music_dirs)
    if value not in music_dirs:
        music_dirs.append(value)
    updated = TonepathConfig(
        music_dirs=tuple(music_dirs),
        data_dir=current.data_dir,
        player=current.player,
        network_mode=current.network_mode,
        privacy=current.privacy,
    )
    write_config(updated)
    return updated


def render_config(config: TonepathConfig) -> str:
    """Render Tonepath config as a small TOML document."""

    music_dirs = ", ".join(quote_string(path) for path in config.music_dirs)
    return "\n".join(
        [
            f"music_dirs = [{music_dirs}]",
            f"data_dir = {quote_string(config.data_dir)}",
            f"player = {quote_string(config.player)}",
            f"network_mode = {quote_string(config.network_mode)}",
            "",
            "[privacy]",
            f"send_to_llm = {toml_bool(config.privacy.send_to_llm)}",
            f"store_play_history = {toml_bool(config.privacy.store_play_history)}",
            "",
        ]
    )


def quote_string(value: str) -> str:
    """Quote a string for the small TOML subset Tonepath writes."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def toml_bool(value: bool) -> str:
    """Render a TOML boolean."""

    return "true" if value else "false"


def data_dir() -> Path:
    """Return the local Tonepath data directory."""

    override = os.environ.get("TONEPATH_HOME")
    if override:
        return Path(override).expanduser()
    return load_config().expanded_data_dir()


def db_path() -> Path:
    """Return the SQLite database path."""

    return data_dir() / DB_FILENAME


def ensure_data_dir() -> Path:
    """Create and return the local data directory."""

    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
