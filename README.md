# Tonepath

Tonepath v0 is a local-first terminal music state-transition agent.

It scans your local music library, plans a listening path from your current state to a target state, plays tracks through `mpv`, adapts to feedback during the session, and stores preferences locally with full inspect/export/delete controls.

Tonepath does not stream platform audio, scrape Spotify/Kugou/NetEase URLs, run a public radio, generate music, or claim therapeutic effects.

## Repository

- Local path: `/Users/von/Projects/music-agents/tonepath`
- Remote: `https://github.com/SleepyLGod/Tonepath.git`
- License: Apache-2.0 for Tonepath source code.

The Apache-2.0 license applies to this software project only. It does not grant rights to any user's local music library, platform catalog content, generated audio from third-party providers, or metadata governed by external platform terms.

## Current Scope

- Local music scanning first.
- Deterministic state-transition planning.
- SQLite profile, feedback, play, and session storage.
- Auditable explanations based only on stored metadata, features, phases, and feedback.
- `mpv` playback adapter for local files.
- CLI first, with a TUI entrypoint stub for the product surface.

## Spotify Boundary

Spotify support is intentionally out of v0 playback scope. A future adapter may create playlists or open Spotify URIs for official-client playback. Tonepath will not stream Spotify audio, overlap/mix Spotify content, run non-interactive webcasting, or depend on restricted Spotify audio feature/recommendation endpoints as core functionality.

## Quick Start

```bash
cd /Users/von/Projects/music-agents/tonepath
uv sync
uv run tonepath config init
uv run tonepath config add-music-dir ~/Music
uv run tonepath doctor
uv run python -m unittest discover -s tests
uv run tonepath scan
uv run tonepath start "我现在很烦，想半小时后进入写代码状态，不要人声"
```

`uv sync` creates this project's isolated `.venv` under the repository. Use `uv run ...` from this directory instead of installing Tonepath into a global Python environment. Commit `uv.lock` for reproducible dependency resolution.

Use `--dry-run` with `start` to print the selected path without launching `mpv`.

## Config

Tonepath reads a small local TOML config from:

```text
~/.tonepath/config.toml
```

Set `TONEPATH_HOME` to move both the config and local data directory for a project-local or test run.

```bash
uv run tonepath config init
uv run tonepath config show
uv run tonepath config add-music-dir ~/Music
uv run tonepath scan
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

`tonepath scan <dir>` still scans one explicit directory. `tonepath scan` with no argument scans every configured `music_dirs` entry.

## Project Boundaries

- v0 uses local music files first.
- v0 does not play Spotify audio inside Tonepath.
- v0 does not scrape Kugou, NetEase, Spotify, or other platform audio URLs.
- Future Spotify support is limited to metadata, playlist creation, and URI handoff to the official Spotify client.
- Privacy is local by default.

## Privacy

Tonepath is offline by default. Local data is stored under:

```text
~/.tonepath/tonepath.db
```

Set `TONEPATH_HOME` or `data_dir` in the config to change the data directory. No audio files or full library data are uploaded by v0.
