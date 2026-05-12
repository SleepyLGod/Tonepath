# Tonepath

Tonepath is a local-first terminal music state-transition agent.

It turns a request like `I am irritated and want to focus in 30 minutes` into an explainable listening path: current state -> intermediate phases -> target state. Tonepath scans your local music library, stores metadata locally, selects tracks with confidence labels, and plays local files through `mpv`.

Tonepath is currently a working CLI prototype. It is not yet a full TUI, macOS app, web app, Spotify player, music generator, or therapy product.

## Status

| Area | Status | Notes |
| --- | --- | --- |
| Project setup | Implemented | Project-local `uv` workflow, CLI entrypoint, Apache-2.0 license. |
| Config | Implemented | Local TOML config at `~/.tonepath/config.toml`; `TONEPATH_HOME` can isolate config and data. |
| Library scanning | Implemented | Scans local audio files and reads metadata with Mutagen, falling back to filenames when tags are missing. |
| Storage | Implemented | SQLite stores tracks, sessions, phases, feedback, profile summaries, and future audio feature rows. |
| Path planning | Implemented | Deterministic prompt parsing and phase planning for state transitions such as irritated -> focus. |
| Track selection | Implemented | Deterministic scoring with confidence labels; metadata-only selections are intentionally low confidence. |
| Playback | Implemented | Local `mpv` adapter plus `--dry-run` command preview. |
| CLI commands | Implemented | `doctor`, `config`, `scan`, `start`, `feedback`, `profile`, `privacy`, and `explain`. |
| Explanations | Implemented | Explanations only cite stored metadata, features, phases, and feedback; unknown BPM/vocalness stays unknown. |
| TUI | Placeholder | A basic Textual shell exists; the real product interface is not implemented yet. |
| Tests | Implemented | Unit tests cover planner, scanner, config, privacy, and explanation behavior. |

## Roadmap

| Area | Planned behavior |
| --- | --- |
| TUI | Path timeline, current track, queue, feedback hotkeys, why panel, and privacy badge. |
| Audio analysis | Local BPM, loudness, energy, vocalness, arousal/valence estimates, and confidence scoring. |
| Feedback loop | Make `like`, `skip`, `too-loud`, `too-slow`, and `no-vocals` change the next candidate during a session. |
| Profile learning | Better local profile rules and preference learning that users can inspect, export, and delete. |
| Spotify handoff | Optional playlist creation and URI handoff to the official Spotify client only. |
| App shells | Future macOS app or web remote built on top of the same local-first core. |

## Not In Scope

Tonepath v0 does not:

- play Spotify, Kugou, NetEase, or other platform audio inside Tonepath;
- scrape platform audio URLs;
- run public radio or non-interactive webcasting;
- generate music;
- mix, overlap, or remix platform content;
- make mental health, therapy, or medical claims.

## Requirements

- Python 3.11+
- `uv`
- `mpv` for actual local playback

On macOS, install `mpv` with:

```bash
brew install mpv
```

You can still use `--dry-run` without starting audible playback.

## Quick Start

```bash
git clone https://github.com/SleepyLGod/Tonepath.git
cd Tonepath
uv sync
uv run tonepath config init
uv run tonepath config add-music-dir ~/Music
uv run tonepath doctor
uv run tonepath scan
uv run tonepath start "我现在很烦，想半小时后进入写代码状态，不要人声"
```

Preview the selected path and `mpv` command without playing audio:

```bash
uv run tonepath start "from irritated to focused in 30 minutes, no vocals" --dry-run
```

Scan one explicit directory instead of configured directories:

```bash
uv run tonepath scan /path/to/music
```

## Config

Default config path:

```text
~/.tonepath/config.toml
```

Default config:

```toml
music_dirs = ["~/Music"]
data_dir = "~/.tonepath"
player = "mpv"
network_mode = "offline"

[privacy]
send_to_llm = false
store_play_history = true
```

Useful commands:

```bash
uv run tonepath config show
uv run tonepath config add-music-dir /path/to/music
uv run tonepath doctor
```

Set `TONEPATH_HOME` to use an isolated config and data directory for tests or alternate profiles:

```bash
TONEPATH_HOME=/tmp/tonepath-demo uv run tonepath config init
```

## Data and Privacy

Tonepath is offline by default. v0 does not upload audio files, full library data, or playback history.

Default local data path:

```text
~/.tonepath/tonepath.db
```

Inspect local storage:

```bash
uv run tonepath privacy status
uv run tonepath profile inspect
```

Delete local profile, session, feedback, and play data while keeping scanned tracks:

```bash
uv run tonepath profile delete --all
```

Local test music belongs outside git. The repository ignores `songs/`, `.venv/`, caches, and local database files.

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
