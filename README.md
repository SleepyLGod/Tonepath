# Tonepath

Tonepath is a local-first terminal music state-transition agent.

It turns a request like `I am irritated and want to focus in 30 minutes` into an explainable listening path: current state -> intermediate phases -> target state. Tonepath scans your local music library, stores metadata locally, selects tracks with confidence labels, and can play local files through `mpv`.

Tonepath is currently a working terminal prototype. It is not a macOS app, web app, Spotify player, music generator, or therapy product.

## Status

| Area | Status | Notes |
| --- | --- | --- |
| Project setup | Implemented | Project-local `uv` workflow, CLI entrypoint, Apache-2.0 license. |
| Config | Implemented | Local TOML config at `~/.tonepath/config.toml`; `TONEPATH_HOME` can isolate config and data. |
| Library scanning | Implemented | Scans local audio files and reads metadata with Mutagen, falling back to filenames when tags are missing. |
| Storage | Implemented | SQLite stores tracks, sessions, phases, feedback, profile summaries, future audio feature rows, and source-attributed enrichment fields. |
| Path planning | Implemented | Deterministic prompt parsing and phase planning for state transitions such as irritated -> focus. |
| Track selection | Implemented | Deterministic scoring with confidence labels; metadata-only selections are intentionally low confidence. |
| Playback | Implemented | Local `mpv` adapter plus `--dry-run` command preview. |
| CLI commands | Implemented | `doctor`, `config`, `scan`, `start`, `feedback`, `profile`, `privacy`, `explain`, and `enrich`. |
| Explanations | Implemented | Explanations only cite stored metadata, features, phases, and feedback; unknown BPM/vocalness stays unknown. |
| Feedback loop | Implemented | The session runtime records feedback and updates upcoming candidates for skip, no-vocals, too-loud, too-slow, and like. |
| TUI | MVP | Textual screen with timeline, controlled playback, queue, why panel, privacy badge, footer shortcuts, and event log. It does not autoplay on launch. |
| Enrichment | Local scaffold | Local metadata enrichment is available; online providers are opt-in boundaries and do not make requests yet. |
| Audio analysis | Basic | `tonepath analyze --features basic` stores local feature rows; WAV files get approximate loudness/energy, other formats stay low-confidence partial rows. |
| Tests | Implemented | Unit tests cover planner, scanner, config, privacy, explanation, session feedback, enrichment, and TUI launch behavior. |

## Roadmap

| Area | Planned behavior |
| --- | --- |
| Deep audio analysis | Local BPM, vocalness, arousal/valence estimates, and stronger confidence scoring. |
| TUI polish | More refined timeline, queue interaction, and layout styling after the controlled playback loop is stable. |
| Profile learning | Better local profile rules and preference learning that users can inspect, export, and delete. |
| Online enrichment | MusicBrainz, AcoustID, ListenBrainz, or cited web enrichment as explicit opt-in providers with cache and rate-limit handling. |
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

You can still use `--dry-run` without starting audible playback. Foreground playback can be stopped with `Ctrl+C`.

## Quick Start

```bash
git clone https://github.com/SleepyLGod/Tonepath.git
cd Tonepath
uv sync
uv run tonepath config init
uv run tonepath config add-music-dir ~/Music
uv run tonepath doctor
uv run tonepath scan
uv run tonepath analyze --features basic
uv run tonepath
uv run tonepath tui "我现在很烦，想半小时后进入写代码状态，不要人声"
uv run tonepath start "我现在很烦，想半小时后进入写代码状态，不要人声"
```

The TUI opens with a planned local session but does not autoplay. Playback events are recorded locally in SQLite for future preference learning. Use these keys:

```text
space / p  play current track
x          stop playback
s          skip
l          like
v          no-vocals
+          too-loud
-          too-slow
w          show why
q          stop playback and quit
```

Preview the selected path and `mpv` command without playing audio:

```bash
uv run tonepath start "from irritated to focused in 30 minutes, no vocals" --dry-run
```

Run playback in the background and stop only Tonepath-managed `mpv` later:

```bash
uv run tonepath start "from irritated to focused in 30 minutes" --background
uv run tonepath stop
```

Scan one explicit directory instead of configured directories:

```bash
uv run tonepath scan /path/to/music
```

Store source-attributed local metadata enrichment:

```bash
uv run tonepath enrich --provider local
```

Store local basic feature rows:

```bash
uv run tonepath analyze --features basic
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

## Enrichment Boundaries

Tonepath separates music understanding into explicit tiers:

| Tier | Status | Behavior |
| --- | --- | --- |
| `local` | Implemented | Stores existing local metadata as source-attributed enrichment records. |
| `features` | Basic | Stores local basic analysis rows. WAV files get approximate loudness and energy; BPM/vocalness are still planned. |
| `online` | Planned | Will require explicit opt-in, cache results, cite sources, and avoid sending local file paths. |

Online providers are blocked by default:

```bash
uv run tonepath enrich --provider musicbrainz
```

This exits without making a network request unless future support enables `network_mode = "online"` and explicit confirmation.

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
