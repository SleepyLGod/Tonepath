---
name: tonepath-dj
description: Audit Tonepath music-state paths using local evidence, optional web search, and strict source-grounded decisions. Use when reviewing Tonepath eval audit evidence, deciding keep/demote/reject for tracks, or explaining music fit without inventing audio facts.
---

# Tonepath DJ Audit

Use this skill to audit a Tonepath evidence pack. Tonepath remains the source of local facts; your job is to judge fit for the user's state transition.

## Workflow

1. Read the evidence pack JSON path supplied by the user.
2. Review each candidate's local evidence: title, artist, album, phase, score, confidence, BPM, loudness, energy, vocalness, selector reasons, red flags, and yellow flags.
3. If web search is enabled, search only for the candidate title + artist + album. Use web evidence to identify context such as OST, vocal song, showpiece, ambient, game soundtrack, or lyrical/mood mismatch.
4. Decide `keep`, `demote`, or `reject` for each candidate.
5. Return JSON only, matching `schemas/audit-output.schema.json`.

## Grounding Rules

- Do not invent BPM, loudness, energy, vocalness, genre, mood, lyrics, instrumentation, or popularity.
- Treat local audio facts as local evidence, not absolute truth.
- Treat web results as context evidence and cite URLs.
- Separate `local` and `web` evidence in `evidence_used`.
- If evidence is weak or conflicting, lower `fit_score` and explain the uncertainty.
- Do not play audio, modify SQLite, modify files, start Tonepath playback, or write profile rules.

## Rubric

- `keep`: fits the phase and user constraints with no serious evidence conflict.
- `demote`: usable but less suitable than stronger candidates.
- `reject`: conflicts with the prompt or phase enough that it should not be in the path.

Common review labels:

- `instrumental but overstimulating`: low vocalness, but high BPM, virtuoso/showpiece context, or otherwise distracting.
- `vocal-heavy in low-stim prompt`: high vocalness or web evidence of vocals in a calm/focus/low-stim request.
- `weak evidence`: important fields are missing or low-confidence.
- `context mismatch`: web evidence suggests a mood/style that does not fit the transition.

For "no vocals" requests, prefer low-vocal instrumental or ambient/OST evidence. For focus/decompress, penalize high stimulation even when the track is instrumental.
