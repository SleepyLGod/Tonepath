# Tonepath

Tonepath is a local-first terminal music state-transition agent.

It turns a request like `I am irritated and want to focus in 30 minutes` into an explainable listening path: current state -> intermediate phases -> target state. Tonepath scans your local music library, stores metadata locally, selects tracks with confidence labels, and can play local files through `mpv`.

Tonepath is currently a working terminal prototype. It is not a macOS app, web app, Spotify player, music generator, or therapy product.

## Status

| Area | Status | Notes |
| --- | --- | --- |
| Project setup | Implemented | Project-local `uv` workflow, CLI entrypoint, Apache-2.0 license. |
| Config | Implemented | Local TOML config under the active `TONEPATH_HOME`; the development default is the workspace-local `.tonepath/` directory. |
| Library scanning | Implemented | Scans local audio files and reads metadata with Mutagen, falling back to filenames when tags are missing. |
| Storage | Implemented | SQLite stores tracks, sessions, phases, feedback, profile summaries, future audio feature rows, and source-attributed enrichment fields. |
| Path planning | Implemented | Deterministic bilingual prompt parsing and phase planning for state transitions such as irritated -> focus. |
| Track selection | Implemented | Deterministic scoring with confidence labels; metadata-only selections are intentionally low confidence. |
| Playback | Implemented | Local `mpv` adapter plus `--dry-run` command preview. |
| CLI commands | Implemented | `prepare`, `status`, `doctor`, `config`, `scan`, `start`, `feedback`, `profile`, `privacy`, `explain`, `eval`, and `enrich`. |
| Explanations | Implemented | Explanations only cite stored metadata, features, phases, and feedback; unknown BPM/vocalness stays unknown. |
| Feedback loop | Implemented | The session runtime records feedback and updates upcoming candidates for skip, no-vocals, too-loud, too-slow, and like. |
| TUI | MVP | Textual screen with prompt intake, timeline, controlled playback, queue, why panel, privacy badge, footer shortcuts, and event log. It does not autoplay on launch. |
| Enrichment | Local scaffold | Local metadata enrichment is available; online providers are opt-in boundaries and do not make requests yet. |
| Audio analysis | Basic + optional MIR | `tonepath prepare` runs the normal scan and analysis flow. Advanced users can still call `tonepath analyze` directly. |
| Model runtime | Scaffold | `tonepath models doctor` and `tonepath models setup essentia-tf` manage a workspace-local TensorFlow tagging runtime outside the main `.venv`. |
| LLM | Opt-in scaffold | `tonepath llm doctor` and `tonepath parse --llm` support DeepSeek/Qwen OpenAI-compatible parsing without exposing local paths or audio facts. |
| Tests | Implemented | Unit tests cover planner, scanner, config, privacy, explanation, session feedback, enrichment, and TUI launch behavior. |

## Roadmap

| Area | Planned behavior |
| --- | --- |
| Deep audio analysis | Optional model-backed vocalness, arousal/valence estimates, and stronger confidence scoring. |
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
cp .env.example .env
uv run tonepath config init
uv run tonepath config add-music-dir ~/Music
uv run tonepath doctor
uv run tonepath prepare
uv run tonepath status
uv run tonepath
uv run tonepath tui "我现在很烦，想半小时后进入写代码状态，不要人声"
uv run tonepath start "我现在很烦，想半小时后进入写代码状态，不要人声"
```

The TUI opens as a local workbench. Run `uv run tonepath` or `uv run tonepath tui`, type a listening goal, and press Enter to create a session. Passing a prompt to `tonepath tui "..."` creates the session immediately, but still does not autoplay. Playback events are recorded locally in SQLite for future preference learning.

`tonepath prepare` is the normal user-facing setup command. It scans configured music directories, prunes missing tracks, analyzes missing or changed MIR features, and follows the configured model policy. The default `balanced` policy uses the workspace-local Essentia-TF runtime for vocalness/tagging when that runtime is ready. If the tagging runtime is missing, `prepare` prints the setup command and still leaves the local library usable.

`tonepath status` is the readiness dashboard. It shows library coverage, model policy, runtime readiness, local data path, network mode, and a concrete next action such as `Run tonepath prepare`, `Run tonepath models setup essentia-tf`, or `Ready for TUI`.

Use these keys:

```text
/          focus prompt
n          new prompt
Enter      submit prompt when input is focused
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

Run a short preparation pass for testing:

```bash
uv run tonepath prepare --limit 5
uv run tonepath prepare --fast
uv run tonepath prepare --full
uv run tonepath prepare --full --setup-models
uv run tonepath status
```

## Advanced Analysis

Store source-attributed local metadata enrichment:

```bash
uv run tonepath enrich --provider local
```

Store local basic feature rows:

```bash
uv run tonepath analyze --features basic
uv run tonepath analyze --features vocalness
uv run tonepath analyze --features vocalness --method spectral
```

Optional faster MIR adapter:

```bash
uv sync --extra mir
uv run tonepath analyze --features mir --method essentia --limit 20
```

Essentia MIR stores BPM, loudness, and energy in `track_features`, and stores descriptors such as key, scale, danceability, and dynamic complexity as source-attributed enrichment records. This is the preferred local route for rhythm/tonal/energy features.

Optional slow separation fallback:

```bash
uv sync --extra models
uv run tonepath analyze --features vocalness --method audio-separator --only-missing --limit 20
uv run tonepath analyze --features vocalness --method demucs-cli
```

`audio-separator` is a slow local source-separation fallback, not the primary route for music understanding. It is installed only when you run `uv sync --extra models`, and its first real run may download model files into Tonepath's local cache. Full-song separation can take several minutes per track on CPU or Apple Silicon acceleration, so run it before listening; Tonepath never runs source separation during playback. `demucs-cli` remains available for users who already have a separate `demucs` command on PATH.

If you want both optional stacks in the same project environment, run `uv sync --extra mir --extra models`.

Experimental tagging boundary:

```bash
uv run tonepath analyze --features tags --method essentia --limit 20
```

Tagging is intentionally not advertised as ready. The current Essentia wheel supports MIR extraction on this project, but TensorFlow music-tagging model support is not available in the default environment. When unavailable, Tonepath fails clearly instead of falling back to guessed tags.

Workspace-local TensorFlow tagging runtime:

```bash
uv run tonepath models doctor
uv run tonepath models setup essentia-tf
uv run tonepath analyze --features tags --method essentia-tf --limit 20
```

The setup command creates a separate Python 3.11 runtime under `TONEPATH_HOME/runtimes/essentia-tf-py311/` and downloads Essentia model files under `TONEPATH_HOME/cache/models/essentia/`. This keeps the main Tonepath environment clean. Playback and TUI never run tagging models in real time; they only read stored SQLite evidence.

Optional LLM prompt parsing:

```bash
uv run tonepath llm doctor
uv run tonepath parse --llm "我现在很烦，想半小时后进入写代码状态，不要人声"
```

LLM parsing uses DeepSeek or Qwen API keys from environment variables or `.env`. It only parses user intent; it must not invent BPM, vocalness, genre, artist metadata, or other audio facts.

Model analysis is resumable and incremental:

```bash
uv run tonepath analyze --features vocalness --method audio-separator --only-missing
uv run tonepath analyze --features vocalness --method audio-separator --changed-only
uv run tonepath analyze --features vocalness --method audio-separator --force --limit 5
```

By default, model methods skip existing results from the same method. Use `--force` to recompute. Use `--limit` for small batches and rerun with `--only-missing` after an interruption.

Evaluate selection quality without playback or profile writes:

```bash
uv run tonepath eval intent
uv run tonepath eval intent --json
uv run tonepath eval selection "我现在很烦，想半小时后进入写代码状态，不要人声" --limit 8
uv run tonepath eval selection "我现在很烦，想半小时后进入写代码状态，不要人声" --json
uv run tonepath eval suite --limit 5
uv run tonepath eval suite --json
uv run tonepath eval audit "我现在很烦，想半小时后进入写代码状态，不要人声" --json
uv run tonepath eval audit "我现在很烦，想半小时后进入写代码状态，不要人声" --codex --web --limit 12
uv run tonepath eval rerank "我现在很烦，想半小时后进入写代码状态，不要人声" --latest
```

`eval intent` checks the packaged Chinese/English prompt-intent fixture corpus. Tonepath uses a deterministic bilingual parser as its local baseline; public corpora such as MASSIVE, GoEmotions, Chinese emotion lexicons, MusicCaps, and MTG-Jamendo are useful references for vocabulary and test design, but Tonepath does not download or vendor those datasets at runtime.

`eval suite` runs a small built-in set of product prompts and flags likely quality problems such as high vocalness in no-vocals results, high stimulation in focus/decompress phases, or low-evidence top candidates. It is read-only: it does not create sessions, playback rows, feedback, or profile rules.

`eval audit` writes a local evidence pack under `TONEPATH_HOME/cache/audit/`. With `--codex`, Tonepath invokes Codex in read-only mode against the packaged Tonepath DJ audit skill in `src/tonepath/resources/codex/skills/tonepath-dj/`. With `--web`, Codex may use web search for cited context. Codex audit is opt-in and does not play audio or mutate the database.

`eval rerank --latest` reads the newest Codex audit result whose evidence prompt matches the current prompt, then prints an advisory queue: `keep` stays in order, `demote` moves later, `reject` is excluded from the suggested queue but still shown with its reason, and unaudited candidates keep their original order. It is read-only and does not change selector weights, playback queues, sessions, feedback, or profile data.

## Config

Development default config path:

```text
/Users/von/Projects/music-agents/.tonepath/config.toml
```

Default config:

```toml
music_dirs = ["~/Music"]
data_dir = "/Users/von/Projects/music-agents/.tonepath"
player = "mpv"
network_mode = "offline"

[privacy]
send_to_llm = false
store_play_history = true

[models]
mode = "balanced"
allow_setup = false
allow_online = false
preferred_tagger = "essentia-tf"
separator_fallback = "off"
```

Useful commands:

```bash
uv run tonepath config show
uv run tonepath config add-music-dir /path/to/music
uv run tonepath doctor
```

Set `TONEPATH_HOME` to use an isolated config and data directory for tests or alternate profiles. The repo-local `.env` may also define this value for development:

```bash
TONEPATH_HOME=/tmp/tonepath-demo uv run tonepath config init
```

## Data and Privacy

Tonepath is offline by default. v0 does not upload audio files, full library data, or playback history.

Development local data path:

```text
/Users/von/Projects/music-agents/.tonepath/tonepath.db
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

Local test music and secrets belong outside git. The repository ignores `songs/`, `.venv/`, `.env`, caches, and local database files. Commit `.env.example`, never `.env`.

## Enrichment Boundaries

Tonepath separates music understanding into explicit tiers:

| Tier | Status | Behavior |
| --- | --- | --- |
| `local` | Implemented | Stores existing local metadata as source-attributed enrichment records. |
| `features` | Basic + optional MIR | Stores local analysis rows. WAV, MP3, FLAC, and M4A can get approximate loudness, energy, conservative BPM, and spectral vocalness when decodable. Optional Essentia MIR can add stronger BPM/loudness/key/danceability descriptors. |
| `online` | Planned | Will require explicit opt-in, cache results, cite sources, and avoid sending local file paths. |

Optional model-backed analysis remains local:

| Method | Status | Behavior |
| --- | --- | --- |
| `spectral` | Default | Lightweight local vocalness proxy. No model download and no network access. |
| `essentia` | Optional MIR | Uses the `mir` extra for offline rhythm, loudness, tonal, and danceability descriptors. |
| `essentia-tf` | Workspace-local tagging runtime | Uses a separate Python 3.11 runtime for Essentia TensorFlow music tagging models. Results are stored as local source-attributed evidence. |
| `audio-separator` | Slow fallback | Uses the `models` extra to run local offline stem separation. Outputs are cached under Tonepath data as `model-audio-separator`, but this is not a fast music-tagging model. |
| `demucs-cli` | Compatibility adapter | Uses a separately installed Demucs CLI to estimate vocalness from the vocal stem. Results are stored as `model-demucs-cli`; this path is for advanced users who already have Demucs installed. |

Model preparation policy lives in config:

| Setting | Default | Behavior |
| --- | --- | --- |
| `models.mode` | `balanced` | `fast` runs scan/MIR only; `balanced` runs tagging when ready; `full` asks for model-backed tagging. |
| `models.allow_setup` | `false` | When true, `prepare` may create the workspace-local model runtime. The CLI flag `--setup-models` enables this per run. |
| `models.allow_online` | `false` | Reserved for future opt-in online identity/LLM workflows. Audio facts stay local. |
| `models.preferred_tagger` | `essentia-tf` | Preferred local tagging runtime for voice/instrumental and music tags. |
| `models.separator_fallback` | `off` | Keeps slow source separation out of the normal user flow unless an advanced user opts in. |

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
