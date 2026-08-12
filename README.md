# Tonepath

[![CI](https://github.com/SleepyLGod/Tonepath/actions/workflows/ci.yml/badge.svg)](https://github.com/SleepyLGod/Tonepath/actions/workflows/ci.yml)

Tonepath is a local-first private radio agent for state-transition listening.

Tell it how you feel and where you want to land. Tonepath turns a request like `I am irritated and want to focus in 30 minutes` into an explainable listening path: current state -> intermediate phases -> target state. It scans your local music library, stores audio evidence locally, selects tracks with confidence labels and reasons, and can play local files through `mpv`.

![Tonepath TUI showing an explainable local listening path](docs/assets/tonepath-tui.png)

Tonepath TUI: prompt, state-transition path, queue, why panel, playback controls, and local evidence.

Tonepath is currently a working terminal prototype. It is not a macOS app, web app, Spotify player, music generator, or therapy product.

For detailed usage, model setup, evaluation, profile learning, and privacy notes, read the [User Guide](docs/user-guide.md). For product direction, read the [Private Radio Agent Roadmap](docs/tonepath-private-radio-agent-roadmap.md).

## What It Does

- prepares a local music library with source-attributed metadata and audio evidence;
- guides first-time setup through Music, Experience, and Review without storing API keys;
- turns Chinese or English listening goals into state-transition phases;
- selects a short local listening path with confidence, reasons, and low-stimulation safety checks;
- keeps replayable local session history with bookmarks and JSON/M3U8 export bundles;
- remembers reversible Like/Dislike reactions and keeps contextual Skip feedback separate;
- inventories, exports, and selectively deletes local personal data through explicit privacy previews;
- keeps optional model, LLM, Codex, and online workflows explicit and opt-in.

## Status

| Area | Status | Notes |
| --- | --- | --- |
| CLI workflow | Implemented | Guided `setup`, plus `prepare`, `status`, `listen`, `history`, `profile`, `privacy`, `eval`, and related commands. |
| Local library | Implemented | Scans local audio metadata, stores tracks/features/sessions/feedback in SQLite under `TONEPATH_HOME`. |
| Planning | Implemented | Deterministic Chinese/English prompt parsing and state-transition phase planning. |
| Selection | Implemented | Explainable deterministic scoring with confidence labels, profile-rule support, low-stimulation semantic safety, and gentle-uplift affect handling. |
| Playback | Implemented | Local `mpv` adapter with foreground, background, stop, and `--dry-run` preview. |
| TUI | MVP | Textual workbench for prompt intake, playback, reactions, disliked-track preview, history, memory, privacy, and setup. |
| Audio analysis | Basic + optional models | Default lightweight analysis plus optional MIR, Essentia-TF tagging/affect, experimental CLAP bake-off, and separator fallback paths. |
| Profile learning | Early-stage | Stable reactions, contextual feedback, suggestions, Markdown memory/evidence, rule apply, and profile comparison are available. |
| Data & Privacy | CLI + TUI implemented | Five-category local inventory, sanitized export, and typed preview-before-delete controls. |
| LLM/Codex | Opt-in | Used for intent parsing, audit, and profile suggestions only when explicitly invoked. |

## Requirements

- Python 3.11+
- `uv`
- `mpv` for audible local playback

On macOS:

```bash
brew install mpv
```

You can still use `--dry-run` and evaluation commands without playing audio.

## Quick Start

```bash
git clone https://github.com/SleepyLGod/Tonepath.git
cd Tonepath
uv sync
cp .env.example .env
uv run tonepath setup
uv run tonepath prepare
uv run tonepath status
uv run tonepath listen "from tired to focused in 30 minutes, no vocals" --dry-run
```

The normal workflow is:

```text
setup -> prepare -> listen or tonepath
```

`setup` asks for your music folder, a Private or Smart experience, and a final review. It can prepare the library after a separate confirmation. `status` tells you whether the library is ready and what to do next. `listen` is the smart default CLI path; `tonepath` opens the TUI workbench.

## Common Commands

```bash
uv run tonepath doctor
uv run tonepath setup --preset smart --dry-run
uv run tonepath status
uv run tonepath prepare --limit 5
uv run tonepath listen "from irritated to focused in 30 minutes, no vocals" --dry-run
uv run tonepath eval suite --limit 5
uv run tonepath eval diagnose --limit 8
uv run tonepath history list
uv run tonepath feedback reactions
uv run tonepath profile inspect
uv run tonepath privacy inspect
```

Actual playback uses local files through `mpv`:

```bash
uv run tonepath listen "from tired to energized in 10 minutes"
uv run tonepath stop
```

## Documentation

- [User Guide](docs/user-guide.md): full CLI workflow, TUI keys, analysis/model setup, evaluation, profile learning, privacy, and config.
- [Private Radio Agent Roadmap](docs/tonepath-private-radio-agent-roadmap.md): product thesis, gaps, phases, benchmark strategy, and future direction.

## Not In Scope

Tonepath v0 does not:

- play Spotify, Kugou, NetEase, or other platform audio inside Tonepath;
- scrape platform audio URLs;
- run public radio or non-interactive webcasting;
- generate music;
- mix, overlap, or remix platform content;
- make mental health, therapy, or medical claims.

## Development

Use the project-local environment:

```bash
uv sync
uv run python -m unittest discover -s tests
uv run tonepath doctor
uv run tonepath --help
```

Do not install Tonepath into a global Python environment for development. Use `uv run ...` from the repository so commands use the project-local `.venv`.

## License

Tonepath source code is licensed under Apache-2.0.

The license covers this software project only. It does not grant rights to user music libraries, third-party platform catalogs, generated audio from external providers, or metadata governed by external platform terms.
