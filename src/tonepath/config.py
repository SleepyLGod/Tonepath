"""Configuration helpers for local Tonepath state."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


APP_DIR_NAME = ".tonepath"
CONFIG_FILENAME = "config.toml"
DB_FILENAME = "tonepath.db"
EXPERIENCE_PRESETS = {"private", "smart", "custom"}


def repo_root() -> Path:
    """Return the local Tonepath repository root."""

    return Path(__file__).resolve().parents[2]


def workspace_default_home() -> Path:
    """Return the workspace-local default Tonepath home."""

    root = repo_root()
    if root.name == "tonepath" and (root / "pyproject.toml").exists():
        return root.parent / APP_DIR_NAME
    return Path.home() / APP_DIR_NAME


def local_env_path() -> Path:
    """Return the repo-local dotenv path."""

    return repo_root() / ".env"


def load_local_env(path: Path | None = None) -> None:
    """Load local dotenv values without overriding process environment."""

    env_path = path or local_env_path()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, unquote_env_value(value.strip()))


def unquote_env_value(value: str) -> str:
    """Return a dotenv value with one optional quote layer removed."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


load_local_env()


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy defaults from the local config file."""

    send_to_llm: bool = False
    store_play_history: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model preparation policy from the local config file."""

    mode: str = "balanced"
    allow_setup: bool = False
    allow_online: bool = False
    preferred_tagger: str = "essentia-tf"
    separator_fallback: str = "off"


@dataclass(frozen=True)
class ExperienceConfig:
    """Normal-user experience mode from the local config file."""

    mode: str = "private"


@dataclass(frozen=True)
class TonepathConfig:
    """User-editable local Tonepath configuration."""

    music_dirs: tuple[str, ...]
    data_dir: str
    player: str
    network_mode: str
    privacy: PrivacyConfig
    models: ModelConfig
    experience: ExperienceConfig

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
    return workspace_default_home()


def config_path() -> Path:
    """Return the local TOML config path."""

    return app_home() / CONFIG_FILENAME


def default_config() -> TonepathConfig:
    """Return the default local-first Tonepath configuration."""

    default_root = os.environ.get("TONEPATH_HOME", str(workspace_default_home()))
    return TonepathConfig(
        music_dirs=("~/Music",),
        data_dir=default_root,
        player="mpv",
        network_mode="offline",
        privacy=PrivacyConfig(),
        models=ModelConfig(),
        experience=ExperienceConfig(),
    )


def load_config() -> TonepathConfig:
    """Load the local TOML config, returning defaults when it does not exist."""

    path = config_path()
    defaults = default_config()
    if not path.exists():
        return defaults

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    privacy_data = data.get("privacy", {})
    models_data = data.get("models", {})
    experience_data = data.get("experience", {})
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
        models=ModelConfig(
            mode=str(models_data.get("mode", defaults.models.mode)),
            allow_setup=bool(models_data.get("allow_setup", defaults.models.allow_setup)),
            allow_online=bool(models_data.get("allow_online", defaults.models.allow_online)),
            preferred_tagger=str(models_data.get("preferred_tagger", defaults.models.preferred_tagger)),
            separator_fallback=str(models_data.get("separator_fallback", defaults.models.separator_fallback)),
        ),
        experience=ExperienceConfig(
            mode=str(experience_data.get("mode", defaults.experience.mode)),
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
        models=current.models,
        experience=current.experience,
    )
    write_config(updated)
    return updated


def preset_config(
    preset: str,
    music_dir: Path | None = None,
    allow_model_setup: bool | None = None,
    send_to_llm: bool | None = None,
) -> TonepathConfig:
    """Return a config for one normal-user experience preset."""

    mode = preset.strip().lower()
    if mode not in EXPERIENCE_PRESETS:
        raise ValueError("preset must be one of: private, smart, custom")
    current = load_config()
    music_dirs = current.music_dirs
    if music_dir is not None:
        value = str(music_dir.expanduser())
        music_dirs = (value,)
    if mode == "private":
        privacy = PrivacyConfig(send_to_llm=False, store_play_history=current.privacy.store_play_history)
        models = ModelConfig(
            mode="balanced",
            allow_setup=bool(allow_model_setup) if allow_model_setup is not None else False,
            allow_online=False,
            preferred_tagger=current.models.preferred_tagger,
            separator_fallback="off",
        )
        network_mode = "offline"
    elif mode == "smart":
        privacy = PrivacyConfig(send_to_llm=True if send_to_llm is None else send_to_llm, store_play_history=current.privacy.store_play_history)
        models = ModelConfig(
            mode="full",
            allow_setup=bool(allow_model_setup) if allow_model_setup is not None else False,
            allow_online=True,
            preferred_tagger=current.models.preferred_tagger,
            separator_fallback="off",
        )
        network_mode = "online-opt-in"
    else:
        privacy = PrivacyConfig(
            send_to_llm=current.privacy.send_to_llm if send_to_llm is None else send_to_llm,
            store_play_history=current.privacy.store_play_history,
        )
        models = ModelConfig(
            mode=current.models.mode,
            allow_setup=current.models.allow_setup if allow_model_setup is None else allow_model_setup,
            allow_online=current.models.allow_online,
            preferred_tagger=current.models.preferred_tagger,
            separator_fallback=current.models.separator_fallback,
        )
        network_mode = current.network_mode
    return TonepathConfig(
        music_dirs=music_dirs,
        data_dir=current.data_dir,
        player=current.player,
        network_mode=network_mode,
        privacy=privacy,
        models=models,
        experience=ExperienceConfig(mode=mode),
    )


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
            "[models]",
            f"mode = {quote_string(config.models.mode)}",
            f"allow_setup = {toml_bool(config.models.allow_setup)}",
            f"allow_online = {toml_bool(config.models.allow_online)}",
            f"preferred_tagger = {quote_string(config.models.preferred_tagger)}",
            f"separator_fallback = {quote_string(config.models.separator_fallback)}",
            "",
            "[experience]",
            f"mode = {quote_string(config.experience.mode)}",
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
