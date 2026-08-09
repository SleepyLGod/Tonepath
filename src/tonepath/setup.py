"""Shared configuration draft and review helpers for Tonepath setup."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from tonepath import config


@dataclass(frozen=True)
class SetupDraft:
    """Mutable-by-replacement user choices before setup is saved."""

    music_dirs: tuple[str, ...]
    experience_mode: str
    model_mode: str
    allow_model_setup: bool
    send_to_llm: bool
    store_play_history: bool
    llm_provider: str

    @classmethod
    def from_config(cls, settings: config.TonepathConfig) -> SetupDraft:
        """Build a setup draft from current persisted settings."""

        return cls(
            music_dirs=tuple(str(Path(path).expanduser()) for path in settings.music_dirs),
            experience_mode=settings.experience.mode,
            model_mode=settings.models.mode,
            allow_model_setup=settings.models.allow_setup,
            send_to_llm=settings.privacy.send_to_llm,
            store_play_history=settings.privacy.store_play_history,
            llm_provider=settings.llm.provider,
        )

    def replace_music_dirs(self, paths: tuple[str, ...]) -> SetupDraft:
        """Return a draft with an explicit ordered music directory list."""

        if any(not path.strip() for path in paths):
            raise ValueError("Music directory cannot be empty.")
        normalized = tuple(dict.fromkeys(str(Path(path).expanduser()) for path in paths))
        return replace(self, music_dirs=normalized)

    def add_music_dir(self, path: Path) -> SetupDraft:
        """Return a draft with one directory added without removing others."""

        value = str(path.expanduser())
        if value in self.music_dirs:
            return self
        return replace(self, music_dirs=(*self.music_dirs, value))

    def remove_music_dir(self, path: Path) -> SetupDraft:
        """Return a draft without one directory."""

        value = str(path.expanduser())
        return replace(self, music_dirs=tuple(item for item in self.music_dirs if item != value))

    def with_experience(self, mode: str, *, send_to_llm: bool, provider: str) -> SetupDraft:
        """Return a draft with one user-facing experience and explicit AI consent."""

        normalized_mode = mode.strip().lower()
        if normalized_mode not in config.EXPERIENCE_PRESETS:
            raise ValueError("experience must be one of: private, smart, custom")
        normalized_provider = config.normalize_llm_provider(provider)
        model_mode = self.model_mode
        if normalized_mode == "private":
            model_mode = "balanced"
        elif normalized_mode == "smart":
            model_mode = "full"
        return replace(
            self,
            experience_mode=normalized_mode,
            model_mode=model_mode,
            send_to_llm=False if normalized_mode == "private" else bool(send_to_llm),
            llm_provider=normalized_provider,
        )

    def with_models(self, mode: str, *, allow_setup: bool) -> SetupDraft:
        """Return a draft with local model preparation choices."""

        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"fast", "balanced", "full"}:
            raise ValueError("model mode must be one of: fast, balanced, full")
        return replace(self, model_mode=normalized_mode, allow_model_setup=bool(allow_setup))

    def with_local_history(self, enabled: bool) -> SetupDraft:
        """Return a draft with local playback history enabled or disabled."""

        return replace(self, store_play_history=bool(enabled))

    def to_config(self, base: config.TonepathConfig) -> config.TonepathConfig:
        """Build final config while preserving settings outside setup's ownership."""

        if self.experience_mode == "private":
            network_mode = "offline"
            allow_online = False
        elif self.experience_mode == "smart":
            network_mode = "online-opt-in"
            allow_online = True
        else:
            network_mode = base.network_mode
            allow_online = base.models.allow_online
        return replace(
            base,
            music_dirs=self.music_dirs,
            network_mode=network_mode,
            privacy=config.PrivacyConfig(
                send_to_llm=False if self.experience_mode == "private" else self.send_to_llm,
                store_play_history=self.store_play_history,
            ),
            models=replace(
                base.models,
                mode=self.model_mode,
                allow_setup=self.allow_model_setup,
                allow_online=allow_online,
            ),
            experience=config.ExperienceConfig(mode=self.experience_mode),
            llm=config.LlmConfig(provider=config.normalize_llm_provider(self.llm_provider)),
        )


def validate_music_directories(paths: tuple[str, ...]) -> tuple[Path, ...]:
    """Validate that setup has at least one existing local directory."""

    if not paths:
        raise ValueError("At least one music directory is required.")
    validated: list[Path] = []
    for raw_path in paths:
        if not raw_path.strip():
            raise ValueError("Music directory cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise ValueError(f"Music directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Music directory is not a directory: {path}")
        validated.append(path)
    return tuple(validated)


def setup_review(draft: SetupDraft, *, model_ready: bool, provider_key_ready: bool) -> str:
    """Return a concise human-readable review of unsaved setup choices."""

    directory_lines = "\n".join(f"  - {path}" for path in draft.music_dirs) or "  - none"
    provider_label = "DeepSeek" if draft.llm_provider == "deepseek" else "Qwen"
    if draft.experience_mode == "private" or not draft.send_to_llm:
        ai_line = "AI Assist: off; Request and Memory text stay local"
    elif provider_key_ready:
        ai_line = f"AI Assist: {provider_label} ready; text is sent only for opted-in AI tasks"
    else:
        ai_line = f"AI Assist: {provider_label} key missing; local fallback remains available"
    model_line = "Local models: ready" if model_ready else "Local models: available to set up"
    history_line = "stored locally" if draft.store_play_history else "not stored"
    return "\n".join(
        [
            "Music Library:",
            directory_lines,
            f"Experience: {draft.experience_mode.title()}",
            "Music stays local; Tonepath never uploads audio files.",
            ai_line,
            f"{model_line} ({draft.model_mode})",
            f"Playback history: {history_line}",
        ]
    )


def setup_summary(settings: config.TonepathConfig, *, model_ready: bool, provider_key_ready: bool) -> str:
    """Return the current setup summary shown before selective reconfiguration."""

    review = setup_review(
        SetupDraft.from_config(settings),
        model_ready=model_ready,
        provider_key_ready=provider_key_ready,
    )
    return f"Current Tonepath setup\n{review}"
