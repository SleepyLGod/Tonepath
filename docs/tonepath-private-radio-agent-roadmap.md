# Tonepath Private Radio Agent Roadmap

## Product Thesis

Tonepath is a local-first private radio agent for state-transition listening.

The product should help a listener move from a current state to a target state through an explainable path of local music. It should learn from local listening behavior, explicit feedback, optional LLM reflection, and optional private journaling. Tonepath is not a Spotify clone, a music generator, a therapy product, or a generic AI DJ.

The core loop is:

```text
local library -> audio evidence -> user intent or journal -> path selection -> playback -> feedback -> profile -> better future paths
```

## Current State

Tonepath currently has a working local-first CLI/TUI prototype:

- local TOML config and workspace-local runtime/data paths;
- local library scanning, pruning, dirty metadata reporting, and failed-analysis reporting;
- local audio feature analysis for duration, format, loudness, energy, BPM, vocalness, and tagging where available;
- deterministic bilingual intent parsing for Chinese and English prompts;
- deterministic path planning and selector scoring with explainable reasons;
- controlled `mpv` playback with foreground stop, background stop, and TUI playback controls;
- Textual TUI with prompt intake, timeline, queue, now-playing state, why panel, privacy badge, and event log;
- feedback capture for like, skip, no-vocals, too-loud, and too-slow;
- profile command surface and LLM/Codex-assisted profile suggestions;
- benchmark/eval commands for intent, selection, suite checks, audit packs, Codex audit, and rerank preview;
- optional DeepSeek/Qwen prompt parsing and packaged Codex audit skill;
- GitHub CI and packaged resource checks.

This is enough to prove the product direction. Profile suggestions now exist, but the personalized radio loop is still early-stage: Tonepath must make learned preferences visible, comparable, reversible, and clearly tied to recommendation changes before it can be called a mature personalized music agent.

## Product Gaps

### Listening and Feedback

- Feedback after playback is not yet visible enough in the TUI.
- The user cannot clearly see why the next track changed after a feedback action.
- User profile rules do not yet feel like a first-class product surface.
- `profile inspect` needs to become a readable preference dashboard, not only a technical summary.

### Recommendation Quality

- Selection quality still depends on whether the local library contains suitable music.
- Audio model output, especially vocalness and tags, is useful but not absolute.
- Selector weights need ongoing calibration against real prompts and local libraries.
- Long-term feedback is still underused.
- Transition smoothness between phases is not yet modeled deeply.

### Models and Music Understanding

- There is no formal model bake-off yet.
- Essentia-TF is the main local tagging route, but the quality boundary needs clearer product validation.
- `audio-separator` and Demucs-style separation are useful fallback tools, not primary music-understanding models.
- Future candidates such as YAMNet, MERT, Music2Vec, MuQ/MuLan-style embeddings, or other local music models should be evaluated only after the current profile loop is stable.

### Codex and LLM

- Codex audit is advisory; it does not yet become a controlled recommendation improvement loop.
- Codex is not yet deeply involved in profile maintenance.
- LLM parsing exists, but LLM is not yet a first-class profile reflection engine.
- LLM outputs need strict schemas, validation, and evaluation before they can safely affect profile rules.

### Privacy and Trust

- Privacy boundaries need to be visible in status, TUI, and docs, not only implementation.
- LLM profile evidence packs need continuous privacy tests.
- Users should see what can leave the machine, what stays local, and what evidence supports each suggestion.

## Design Principles

- **Local-first by default.** Audio files, library state, profile rules, journal entries, and playback history stay local unless the user explicitly opts in.
- **LLM is reflective, not factual.** LLMs may summarize preferences, parse language, rewrite explanations, and suggest profile rules. They must not invent BPM, vocalness, genre, mood, lyrics, or track facts.
- **Evidence before recommendation.** Every recommendation should be traceable to prompt intent, stored audio evidence, feedback, profile rules, or clearly labeled audit context.
- **User effort is optional.** A basic user should only need `prepare`, `status`, and `tonepath`. Advanced users can configure models, run audits, inspect profile rules, or write journal entries.
- **No hidden heavy jobs.** TUI must not silently download models, run source separation, or call online services.
- **Advisory before mutation.** LLM and Codex outputs should create pending suggestions first. Applying profile rules should be explicit or governed by a clear local policy.
- **Clean boundaries.** Analysis extracts facts. Planning parses intent. Selection scores candidates. Profile summarizes user preference. LLM reflects over safe evidence. TUI displays and captures input.
- **Privacy is a product surface.** Privacy status should be inspectable and understandable by normal users.

## Near-Term Roadmap

### Phase 1: Profile Learning and Feedback Visibility

Goal: turn feedback into visible, inspectable, and useful user preference.

Key work:

- Generate deterministic profile suggestions from local feedback.
- Generate optional LLM/Codex profile suggestions from privacy-safe evidence packs.
- Require explicit apply before suggestions affect selection.
- Make `profile inspect` show active rules, pending suggestions, source, confidence, and rationale.
- Let selector apply profile rules with clear candidate reasons.
- Add `eval selection --with-profile` and `eval selection --no-profile` to compare behavior.
- Show in TUI when feedback changed the upcoming queue or profile suggestions.

Acceptance:

- Repeated `too-loud` in focus sessions reduces future loudness preference for focus.
- Likes on low-vocal instrumental tracks raise similar candidates in compatible prompts.
- Skips on high-BPM focus tracks demote similar high-stimulation candidates.
- Candidate explanations clearly show profile rule effects.
- LLM/Codex suggestions never upload audio, local paths, API keys, or full library dumps.
- A user can give feedback, inspect what Tonepath learned, apply or delete a rule, and compare selection with and without profile influence.

### Phase 2: Private Journal and Tree-Hole Workflow

Goal: let users express mood and context naturally, without manually tuning preferences.

Journal is a private listening-context surface, not therapy, medical advice, or mental health treatment. It should help Tonepath understand listening intent and constraints without making clinical claims.

Possible commands:

```bash
uv run tonepath journal add
uv run tonepath journal list
uv run tonepath journal reflect --llm --confirm
uv run tonepath journal mood
```

Product behavior:

- Store private journal entries locally.
- Extract state, task, constraints, and music needs from entries.
- Use deterministic parsing by default.
- Use LLM reflection only with explicit opt-in or local privacy policy.
- Feed journal-derived insights into pending profile suggestions, not directly into selector mutation.
- Avoid therapy or medical claims.

Example user entry:

```text
Today my mind is noisy. I do not want vocals. I need a little rhythm, but nothing sharp, so I can write for one hour.
```

Possible derived signal:

- source state: irritated or overloaded;
- target state: focus;
- constraints: avoid vocals, low stimulation;
- preference hypothesis: soft rhythm, low loudness, low vocalness.

Acceptance:

- Journal entries are stored locally and deletable.
- Reflection output is structured and source-grounded.
- Journal text is never sent to LLM without explicit consent.
- TUI can show recent journal-derived intent without turning it into therapy advice.

### Phase 3: Music Understanding and Model Quality

Goal: improve audio evidence quality without turning Tonepath into a model zoo.

Key work:

- Keep Essentia-TF as the current primary local tagging route.
- Treat source separation as slow fallback evidence.
- Build a small model bake-off only when benchmark or real usage shows a concrete weakness.
- Compare models on vocal/instrumental, low stimulation, mood/theme, instrument, and focus suitability.
- Keep all heavy models optional and workspace-local.

Model candidates to evaluate later:

- Essentia music extractors and TensorFlow models;
- YAMNet for broad audio event tags;
- MERT or Music2Vec-style music representation models;
- MuQ/MuLan-style text-music embedding models;
- source separation tools only for vocal stem evidence.

Acceptance:

- New model routes must improve at least one measured failure mode.
- Do not add a new model route without a benchmarked failure mode and measured improvement.
- Every model field stores source and confidence.
- Selector never treats model predictions as absolute truth.
- Playback and TUI never trigger model analysis in real time.

### Phase 4: Product Shells and Integrations

Goal: add broader surfaces only after the core private-radio loop is stable.

Future features:

- Spotify playlist or URI handoff, not Tonepath-controlled Spotify playback.
- macOS app shell over the same local core.
- Web remote for local playback control.
- Richer TUI with profile, journal, and audit panels.
- Playlist export.
- Calendar or task context, if privacy boundaries are clear.
- LLM narrator or session reflection, if it only rewrites stored evidence.

Not now:

- Spotify, Kugou, NetEase, or platform scraping.
- AI music generation.
- Public radio or webcasting.
- Large app shell before the core loop is stable.
- Default online search for every song.

## Benchmark Strategy

Benchmarking should be useful, not performative.

Short term:

- Keep intent parser fixtures for Chinese and English.
- Keep selection suite scenarios for core states such as focus, calm, sleep, energize, and low stimulation.
- Add small no-profile versus deterministic-profile versus LLM/Codex-profile checks.
- Use top-k red/yellow flags to catch obvious bad recommendations.
- Use Codex/web audit as optional spot review, not a mandatory test dependency.

Not now:

- Cross-user benchmark.
- Public leaderboard.
- Large-scale recommender metrics.
- Training-like evaluation loops.

Profile benchmark should stay lightweight:

- Seed a small local feedback pattern.
- Confirm selector movement is in the expected direction.
- Confirm explanations cite profile rules.
- Confirm deleting profile data removes the personalization effect.
- Confirm LLM/Codex-suggested profile rules improve or explain a concrete selection change before treating them as useful.

## LLM Strategy

LLM should participate in:

- natural-language intent parsing;
- journal reflection;
- profile preference summarization;
- profile rule suggestion;
- explanation rewriting;
- Codex audit and rerank review;
- wording of user-facing insights.

LLM should not:

- infer audio facts;
- inspect audio files;
- receive absolute local paths;
- receive API keys or secrets;
- receive full library dumps by default;
- directly mutate SQLite state;
- become the selector's hidden ranking engine.

The safe pattern is:

```text
local evidence pack -> LLM/Codex suggestion -> validation -> pending suggestion -> explicit apply -> selector reason
```

## Engineering Boundaries

Keep modules simple and bounded:

- `scanner.py`: discover local tracks and read metadata.
- `analysis.py`: extract audio facts and model-derived features.
- `planner.py`: parse intent and build state-transition phases.
- `selector.py`: score candidates from evidence, feedback, and applied profile rules.
- `profile.py`: build profile evidence, generate suggestions, and apply profile rules.
- `journal.py`: store and reflect on private journal entries.
- `llm.py`: provider configuration and strict request/response handling.
- `evaluation.py`: read-only audit, benchmark, and comparison flows.
- `tui.py`: display, input, and playback controls only.
- `db.py`: persistence only, not product logic.

If a module needs "and" to describe its responsibility, split it.

Engineering rules:

- Do not add abstractions before a second real use case exists.
- Do not let LLM response parsing become ad hoc string handling.
- Do not add model dependencies to the default install path.
- Do not let TUI hide long-running work.
- Do not let profile learning silently rewrite user state.
- Keep privacy tests near every new evidence pack.

## Immediate Next Step

Validate the full private-radio loop with real listening before starting the journal workflow:

```bash
uv run tonepath status
uv run tonepath eval diagnose --limit 8
uv run tonepath listen "我有点难过，想慢慢开心一点，但不要太吵" --dry-run
uv run tonepath listen "我很焦虑，想二十分钟后平静下来，不要压抑" --dry-run
uv run tonepath profile inspect
uv run tonepath eval profile "我要写论文，四十五分钟，低刺激，最好不要人声" --limit 8
uv run tonepath profile delete --all
```

The current selector path should stay local and evidence-first: low-stimulation semantic risks are demoted, gentle-uplift phases can become slightly brighter, and CLAP/hybrid remain evaluation tools until they show a stable measured improvement. The next product phase should start only after real listening confirms that readiness, library hygiene, selection reasons, feedback, profile inspection, and profile deletion all feel understandable and reversible.
