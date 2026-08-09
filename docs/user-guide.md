# Tonepath User Guide

This guide covers daily usage, advanced analysis, evaluation, profile learning, privacy, and local runtime configuration. The short project overview stays in the main [README](../README.md).

## Normal Workflow

The normal user flow is:

```text
config add-music-dir -> prepare -> status -> tonepath
```

Start from a project-local environment:

```bash
uv sync
cp .env.example .env
uv run tonepath setup --preset private
uv run tonepath config add-music-dir ~/Music
uv run tonepath prepare
uv run tonepath status
uv run tonepath listen "我现在很烦，想写论文，低刺激，不要人声" --dry-run
```

`tonepath setup` offers three normal-user experience presets. `Private` is local-first and offline by default. `Smart` enables opt-in LLM/profile reflection when API keys are configured, while still avoiding silent profile-rule changes. `Custom` marks the config for advanced tuning while preserving existing safety defaults.

`tonepath prepare` scans configured music directories, prunes missing tracks, analyzes missing or changed features, and follows the configured model policy. The default `balanced` policy uses the workspace-local Essentia-TF runtime for vocalness/tagging when that runtime is ready. If the tagging runtime is missing, `prepare` prints the setup command and still leaves the local library usable.

`tonepath status` is the readiness dashboard. It shows an overall state such as `Ready for TUI`, `Needs preparation`, `Review files`, or `Model setup available`, plus a concrete next action. If a local file cannot be analyzed, Tonepath lists it under `Missing analysis files`; replace or remove the file if you want full-confidence recommendations, or keep it and Tonepath will treat it as low evidence.

If recommendations look confusing because of bad tags, inspect the library hygiene report:

```bash
uv run tonepath library issues
uv run tonepath library issues --json
```

Use local display overrides for bad metadata without editing audio files:

```bash
uv run tonepath library set-meta 42 --title "A Better Title" --artist "A Better Artist"
uv run tonepath library clear-meta 42
```

Overrides are stored in Tonepath's SQLite `track_enrichment` table as local metadata evidence. They affect display, duplicate detection, `status`, `listen`, TUI, and evaluation output, but they do not modify scanned raw tags, model caches, or the music files themselves.

`tonepath listen "..."` is the smart default CLI entrypoint. It checks readiness, reports the active experience mode, uses local deterministic planning by default, uses Smart-mode LLM intent parsing only when configured, and then previews or plays a local state-transition path. Long-term preference changes still require explicit `profile inspect` plus `profile apply` or `profile apply-group`.

Run a short preparation pass for testing:

```bash
uv run tonepath prepare --limit 5
uv run tonepath prepare --fast
uv run tonepath prepare --full
uv run tonepath prepare --full --setup-models
uv run tonepath status
```

Scan one explicit directory instead of configured directories:

```bash
uv run tonepath scan /path/to/music
```

## TUI and Playback

The TUI opens as a local workbench. Run `uv run tonepath` or `uv run tonepath tui`, type a listening goal, and press Enter to create a session. Passing a prompt to `tonepath tui "..."` creates the session immediately, but still does not autoplay. Playback events are recorded locally in SQLite for future preference learning.

The TUI has a small built-in theme pack: `Warmline`, `Midnight`, `High Contrast`, `Solarized Dark`, `Solarized Light`, `Catppuccin Mocha`, `Catppuccin Latte`, `Dracula`, and `Jukebox`. Press `t` to cycle themes; Tonepath stores the choice in `config.toml` under `[ui] theme = "..."`.

These palettes are fixed local Tonepath palettes inspired by mature terminal color systems such as Solarized, Catppuccin, Dracula, and black/green music-player TUIs. Tonepath does not download external theme files and does not require Ghostty, iTerm, oh-my-zsh, or any terminal theme to be installed. Fonts are controlled by your terminal app, not Tonepath. Use a readable monospace font in Ghostty, iTerm, Terminal, or your preferred terminal.

The bottom command bar stays visible while you type, so playback keys do not disappear when the Request input is focused. The Now panel reads position, duration, pause state, and volume from the managed `mpv` process. The pulse meter is a decorative energy visual derived from local features; it is not a real-time audio spectrum.

TUI keys:

```text
/          focus prompt
n          new prompt
Enter      submit prompt when input is focused
space / p  play, pause, or resume
Left/Right seek backward/forward 10 seconds
Up/Down    raise/lower volume by 5%
x          stop playback
s          skip and record negative feedback
>          next track without feedback
<          previous track without feedback
l          like
v          prefer less vocals
+          too loud; lower upcoming energy
-          too slow; raise upcoming energy
m          cycle playback mode
Ctrl+L     open listening history
Ctrl+O     open or hide the memory notes panel (Control + letter o, not zero)
Ctrl+S     save memory locally
Ctrl+Enter save memory and update profile with AI Assist
Ctrl+P     show the consolidated memory profile
Ctrl+G     generate memory-derived suggestions
j / k      move through suggestions
Enter      apply selected suggestion when suggestions are focused
t          cycle visual theme
i          show AI Assist status
e          expand/collapse event log
w          write full why to events
a          show Codex audit command
r          show latest Codex rerank preview
?          show full help panel
q          stop playback and quit when Request is not focused
Ctrl+Q     stop playback and quit at any time
```

Arrow keys remain text-editing controls while the Request or Memory editor is focused, and remain History navigation controls inside Listening History. Volume is preserved while Tonepath switches tracks in the current TUI process, but Player Core v1 does not persist it across app restarts.

Playback modes are TUI-local:

```text
Manual         play the current track and stop when it ends
Continue Path  automatically continue through the generated path
Repeat One     replay the current track
Repeat Path    loop the generated path from the beginning
```

Use `>` when you simply want to hear the next candidate. Use `s` when the current track is a bad fit and Tonepath should remember that feedback.

`AI Assist Off` means Tonepath is using only local deterministic intent parsing for new prompts. `AI Assist Ready: deepseek` or `AI Assist Ready: qwen` means Smart mode can call the configured LLM for intent parsing on new prompts. AI Assist does not judge BPM, vocalness, tags, artist metadata, or other audio facts; those remain local evidence.

Preview the selected path and `mpv` command without playing audio:

```bash
uv run tonepath start "from irritated to focused in 30 minutes, no vocals" --dry-run
```

Run playback in the background and stop only Tonepath-managed `mpv` later:

```bash
uv run tonepath start "from irritated to focused in 30 minutes" --background
uv run tonepath stop
```

## Saved Sessions and Exact Replay

Tonepath stores the final queue for every non-dry-run CLI or TUI session. The snapshot keeps the original order, phase, score, confidence, explanation reasons, title, artist, and local path. If feedback rebuilds the future queue, Tonepath updates the snapshot so history reflects what the session actually became.

List sessions that were actually played or explicitly saved:

```bash
uv run tonepath history list
uv run tonepath history list --saved-only
uv run tonepath history list --all
```

Inspect and bookmark a useful path:

```bash
uv run tonepath history show 42
uv run tonepath history save 42 --name "写代码但别太困"
uv run tonepath history unsave 42
```

Exact replay uses the original saved queue and order; it does not call the selector again:

```bash
uv run tonepath history replay 42 --dry-run
uv run tonepath history replay 42
uv run tonepath history replay 42 --background
```

If a file has been removed, replay omits it and names the missing path. Replay stops with a clear error if no playable files remain. Sessions created before queue snapshots were introduced remain viewable, but cannot pretend to support exact replay; `history show` prints a copyable `tonepath listen` command for generating a new path from the original Request.

Export one session for local inspection or another local player:

```bash
uv run tonepath history export 42 --output ~/Desktop/tonepath-session-42
```

The output contains `session.json` and `playlist.m3u8`. The JSON includes the Request, phases, queue evidence, session feedback, and omitted files. Both files may contain absolute local music paths, so keep the bundle on trusted local storage. Tonepath refuses to overwrite a non-empty output directory. History export never includes Memory, API keys, the complete library, or another session.

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

Essentia MIR stores BPM, loudness, and energy in `track_features`, and stores descriptors such as key, scale, danceability, and dynamic complexity as source-attributed enrichment records. This is the preferred local route for rhythm, tonal, and energy features.

Optional slow separation fallback:

```bash
uv sync --extra models
uv run tonepath analyze --features vocalness --method audio-separator --only-missing --limit 20
uv run tonepath analyze --features vocalness --method demucs-cli
```

`audio-separator` is a slow local source-separation fallback, not the primary route for music understanding. It is installed only when you run `uv sync --extra models`, and its first real run may download model files into Tonepath's local cache. Full-song separation can take several minutes per track on CPU or Apple Silicon acceleration, so run it before listening; Tonepath never runs source separation during playback. `demucs-cli` remains available for users who already have a separate `demucs` command on PATH.

If you want both optional stacks in the same project environment:

```bash
uv sync --extra mir --extra models
```

Experimental tagging boundary:

```bash
uv run tonepath analyze --features tags --method essentia --limit 20
```

Tagging is intentionally not advertised as ready in the default environment. When unavailable, Tonepath fails clearly instead of falling back to guessed tags.

Workspace-local TensorFlow tagging runtime:

```bash
uv run tonepath models doctor
uv run tonepath models setup essentia-tf
uv run tonepath analyze --features tags --method essentia-tf --limit 20
uv run tonepath analyze --features affect --method essentia-tf --limit 20
```

The setup command creates a separate Python 3.11 runtime under `TONEPATH_HOME/runtimes/essentia-tf-py311/` and downloads Essentia model files under `TONEPATH_HOME/cache/models/essentia/`. This keeps the main Tonepath environment clean. Playback and TUI never run tagging or affect models in real time; they only read stored SQLite evidence.

The `affect` tier stores arousal and valence in `track_features`, then derives a readable affect profile such as `sadness`, `uplift`, `calmness`, `tension`, `warmth`, `darkness`, and `brightness` from Essentia mood/theme tags plus arousal/valence. These axes are evidence for selector and benchmark logic; they do not replace the raw model tags. Essentia model files are suitable for a local prototype, but their upstream models may carry non-commercial license terms, so do not treat this path as a commercial-ready default without reviewing the upstream licenses.

Experimental music-text bake-off:

```bash
uv run tonepath models setup clap
uv run tonepath analyze --features embedding --method clap --changed-only
uv run tonepath eval bakeoff --engine selector --engine clap --engine hybrid --limit 8
```

The CLAP path is evaluation-only. It creates a separate runtime under `TONEPATH_HOME/runtimes/clap-py311/`, stores model/cache files under `TONEPATH_HOME/cache/models/clap/`, and stores per-track embeddings under `TONEPATH_HOME/cache/embeddings/clap/`. The `hybrid` bake-off engine uses selector-safe candidates first, then adds a bounded CLAP semantic bonus inside that safe pool. It does not change `listen`, TUI playback, selector weights, profile rules, or the SQLite schema. MuQ-MuLan remains a later candidate if CLAP does not improve Chinese or emotion-transition benchmark cases.

Model analysis is resumable and incremental:

```bash
uv run tonepath analyze --features vocalness --method audio-separator --only-missing
uv run tonepath analyze --features vocalness --method audio-separator --changed-only
uv run tonepath analyze --features vocalness --method audio-separator --force --limit 5
```

By default, model methods skip existing results from the same method. Use `--force` to recompute. Use `--limit` for small batches and rerun with `--only-missing` after an interruption.

## How Selection Stays Safe

Tonepath does not treat "quiet" or "instrumental" as automatically good. The selector combines local evidence such as BPM, loudness, energy, vocalness, arousal, valence, and derived affect tags. In low-stimulation phases such as sleep, calm, decompression, and gentle focus, it also applies a semantic safety layer from local metadata and model tags.

That safety layer demotes candidates that look risky for calm listening even when their numeric audio features look acceptable. Examples include march-like, choral or vocal-ensemble, epic, dramatic, ceremonial, anthem-like, showpiece, or very fast/vivace cues. These rules are not hardcoded to specific songs; they are derived from title, artist, genre, filename, and source-attributed tags. When a candidate is demoted this way, its reasons include text such as `semantic risk: march_like for low-stimulation phase`.

For "sad -> gently uplifting" requests, Tonepath uses a `hold -> stabilize -> lift` path. The final `lift` phase gives a small bonus to tracks with higher valence, warmth, or uplift evidence, but only when they are still low-stimulation safe. High energy, high loudness, high vocalness, or semantic-risk candidates do not get this bonus. This is meant to make the path slightly brighter without turning it into a loud or forced happy playlist.

CLAP and `hybrid` remain evaluation tools. They help test whether music-text retrieval can improve semantic ordering, but they do not drive normal `listen` results. The default recommendation path still uses the local selector first because it can enforce no-vocals, low-stimulation, evidence quality, and state-transition constraints more predictably.

## Evaluation and Audit

Evaluate selection quality without playback or profile writes:

```bash
uv run tonepath eval intent
uv run tonepath eval intent --json
uv run tonepath eval selection "我现在很烦，想半小时后进入写代码状态，不要人声" --limit 8
uv run tonepath eval selection "我现在很烦，想半小时后进入写代码状态，不要人声" --json
uv run tonepath eval suite --limit 5
uv run tonepath eval suite --json
uv run tonepath eval bakeoff --engine selector --engine clap --engine hybrid --limit 8
uv run tonepath eval diagnose --limit 8
uv run tonepath eval profile "我要写论文，四十五分钟，低刺激，最好不要人声" --limit 8
uv run tonepath eval audit "我现在很烦，想半小时后进入写代码状态，不要人声" --json
uv run tonepath eval audit "我现在很烦，想半小时后进入写代码状态，不要人声" --codex --web --limit 12
uv run tonepath eval rerank "我现在很烦，想半小时后进入写代码状态，不要人声" --latest
```

`eval intent` checks the packaged Chinese/English prompt-intent fixture corpus. Tonepath uses a deterministic bilingual parser as its local baseline; public corpora such as MASSIVE, GoEmotions, Chinese emotion lexicons, MusicCaps, and MTG-Jamendo are useful references for vocabulary and test design, but Tonepath does not download or vendor those datasets at runtime.

`eval suite` runs a small built-in set of product prompts and flags likely quality problems such as high vocalness in no-vocals results, high stimulation in focus/decompress phases, or low-evidence top candidates. It is read-only: it does not create sessions, playback rows, feedback, or profile rules.

`eval bakeoff` compares the normal selector against optional experimental engines. CLAP uses deterministic English text probes derived from parsed intent so Chinese prompts are evaluated through the same structured intent layer. The `hybrid` engine keeps the selector as the safety layer and only uses CLAP as a small semantic reranker. Bake-off output is advisory only.

`eval diagnose` summarizes why benchmark or bake-off scenarios are failing. It reports root causes such as selector tuning, weak model evidence, library gaps, dirty metadata, benchmark thresholds, or CLAP regressions without printing the full candidate table.

`eval profile` compares selection with and without active profile rules. It is the main way to check whether personalization is helping before changing normal listening behavior.

`eval audit` writes a local evidence pack under `TONEPATH_HOME/cache/audit/`. With `--codex`, Tonepath invokes Codex in read-only mode against the packaged Tonepath DJ audit skill in `src/tonepath/resources/codex/skills/tonepath-dj/`. With `--web`, Codex may use web search for cited context. Codex audit is opt-in and does not play audio or mutate the database.

`eval rerank --latest` reads the newest Codex audit result whose evidence prompt matches the current prompt, then prints an advisory queue: `keep` stays in order, `demote` moves later, `reject` is excluded from the suggested queue but still shown with its reason, and unaudited candidates keep their original order. It is read-only and does not change selector weights, playback queues, sessions, feedback, or profile data.

## Profile Learning and LLM Boundaries

Feedback commands record local preference evidence:

```bash
uv run tonepath feedback like
uv run tonepath feedback skip
uv run tonepath feedback too-loud
uv run tonepath feedback too-slow
uv run tonepath feedback no-vocals
```

Inspect and manage profile learning:

```bash
uv run tonepath profile inspect
uv run tonepath profile suggest
uv run tonepath profile suggest --llm --memory --confirm
uv run tonepath profile suggest --codex --memory
uv run tonepath profile apply <suggestion-id>
uv run tonepath profile apply-group <group-id>
uv run tonepath eval profile "我要写论文，四十五分钟，低刺激，最好不要人声" --limit 8
```

Profile suggestions are pending until explicitly applied. Applied suggestions become local `profile_rules`; selector explanations show profile-rule impact when a rule changes a candidate score. Suggestion groups are a review/apply convenience so complementary rules, such as lower vocalness plus high-BPM demotion, can be applied together.

## Private Memory and Tree-Hole Notes

Memory is separate from ad-hoc listening requests. A request asks Tonepath to make a path now. Memory is where you can paste a short complaint, a long monologue, or recurring listening context that should later shape your profile.

Raw memory entries are appended to a local JSONL log for traceability. The user-facing file is the consolidated Markdown profile:

```text
TONEPATH_HOME/memory/logs/memory-log.jsonl
TONEPATH_HOME/memory/profile.md
```

`memory/profile.md` is the single user-facing Memory Profile shown by both `memory show` and `profile inspect`. It is local context for you and optional LLM/Codex suggestions; it does not directly change selection.

Add memory without changing recommendations:

```bash
uv run tonepath memory add "最近写代码很烦，听到人声会更乱"
uv run tonepath memory add --stdin
```

View or edit the consolidated profile:

```bash
uv run tonepath memory show
uv run tonepath memory edit
```

Update the Markdown profile from new logs with explicit LLM opt-in:

```bash
uv run tonepath memory consolidate --llm --confirm
```

Generate pending profile suggestions from the memory profile, new logs, and feedback evidence:

```bash
uv run tonepath memory suggest --llm --confirm
uv run tonepath profile inspect
uv run tonepath profile apply <suggestion-id>
```

Memory-derived suggestions are advisory until applied. `profile delete --all` clears profile rules, sessions, feedback, profile Markdown/evidence, and pending suggestions, but it does not delete raw memory logs or `memory/profile.md`; those are private user-authored context.

In the TUI, press `Ctrl+O` to open the Memory panel. This means Control + the letter o, not the number 0. Hiding the panel keeps the draft. `Ctrl+S` saves the draft to the local log only. `Ctrl+Enter` saves and updates the Markdown profile when AI Assist is enabled. `Ctrl+P` shows the profile, and `Ctrl+G` shows pending suggestions. Applying a suggestion affects future requests only; it does not stop playback or rewrite the current queue.

Optional LLM prompt parsing:

```bash
uv run tonepath llm doctor
uv run tonepath parse --llm "我现在很烦，想半小时后进入写代码状态，不要人声"
```

LLM parsing uses DeepSeek or Qwen API keys from environment variables or `.env`. It only parses user intent; it must not invent BPM, vocalness, genre, artist metadata, or other audio facts.

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

[experience]
mode = "private"
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

Tonepath is offline by default. Audio files, full library data, Memory, and playback history stay local unless an optional online action is explicitly enabled.

Privacy Center groups local data into five understandable categories:

| Category | What it contains | Privacy Center v1 |
| --- | --- | --- |
| Memory | private notes, `memory/profile.md`, consolidation evidence | inspect, export, delete |
| Personalization | feedback, active rules, profile evidence, pending suggestions | inspect, export, delete |
| Listening History | Requests, paths, queue snapshots, bookmarks, plays, audit evidence | inspect, export, delete |
| Library Evidence | tracks, audio features, enrichment, embeddings, separated evidence | inspect only |
| Models & Storage | model files, isolated runtimes, package caches | inspect only |

Inspect without creating a database or cache:

```bash
uv run tonepath privacy status
uv run tonepath privacy inspect
uv run tonepath privacy inspect --json
uv run tonepath profile inspect
```

`privacy inspect` reports records and known local file sizes. SQLite is shared by several categories, so Tonepath reports the database size once instead of inventing a per-category database size. It does not follow symlinks or scan the size of configured music directories.

Export an owner-only, sanitized personal-data package:

```bash
uv run tonepath privacy export --output ~/tonepath-personal-data
```

The bundle contains Memory, personalization, feedback, listening history, and sanitized settings. It excludes API keys, absolute music paths, raw SQLite, audio files, models, runtimes, and rebuildable caches. Tonepath shows the current external-processing policy, provider, and whether a key exists, but it does not record a transmission history.

Deletion is preview-first. Without `--confirm`, nothing is changed:

```bash
uv run tonepath privacy delete --category memory
uv run tonepath privacy delete --category personalization --category history
uv run tonepath privacy delete --all-personal
```

After reading the exact delete/keep list, rerun the same command with `--confirm`. `--all-personal` removes Memory, Personalization, and Listening History while preserving config, tracks, audio features, model/runtime storage, caches outside those personal categories, and original music.

Deleting Memory by itself keeps active profile rules, which may continue to affect future recommendations. Deleting History while preserving Personalization detaches feedback from deleted sessions. Deletion enables SQLite secure deletion, but Tonepath cannot promise removal from system backups, APFS snapshots, or SSD forensic recovery.

The older command remains available and now uses the same category helpers:

```bash
uv run tonepath profile delete --all
```

It removes Personalization and Listening History, but preserves raw Memory notes and `memory/profile.md` as well as scanned tracks, audio features, model storage, and music files.

Local test music and secrets belong outside git. The repository ignores `songs/`, `.venv/`, `.env`, caches, and local database files. Commit `.env.example`, never `.env`.

## Enrichment Boundaries

Tonepath separates music understanding into explicit tiers:

| Tier | Status | Behavior |
| --- | --- | --- |
| `local` | Implemented | Stores existing local metadata as source-attributed enrichment records. |
| `features` | Basic + optional MIR/affect | Stores local analysis rows. WAV, MP3, FLAC, and M4A can get approximate loudness, energy, conservative BPM, and spectral vocalness when decodable. Optional Essentia MIR can add stronger BPM/loudness/key/danceability descriptors. Optional Essentia-TF affect adds arousal/valence and derived mood axes. |
| `online` | Planned | Will require explicit opt-in, cache results, cite sources, and avoid sending local file paths. |

Optional model-backed analysis remains local:

| Method | Status | Behavior |
| --- | --- | --- |
| `spectral` | Default | Lightweight local vocalness proxy. No model download and no network access. |
| `essentia` | Optional MIR | Uses the `mir` extra for offline rhythm, loudness, tonal, and danceability descriptors. |
| `essentia-tf` | Workspace-local tagging and affect runtime | Uses a separate Python 3.11 runtime for Essentia TensorFlow music tagging and affect models. Results are stored as local source-attributed evidence. |
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
