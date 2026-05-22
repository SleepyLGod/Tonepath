---
name: tonepath-profile
description: Suggest Tonepath user profile rules from local feedback evidence without inventing audio facts or modifying state.
---

# Tonepath Profile Suggestions

Use this skill to read a Tonepath profile evidence pack and propose user preference rules. Tonepath remains the source of local facts; your job is to summarize preference patterns.

## Workflow

1. Read the evidence pack JSON path supplied by the prompt.
2. Review feedback events, prompts, phases, track display metadata, and stored audio features.
3. Propose only rules supported by repeated or clear evidence.
4. Return JSON only, matching `schemas/profile-suggestions.schema.json`.

## Grounding Rules

- Do not invent BPM, loudness, energy, vocalness, genre, mood, artist identity, or track facts.
- Do not use local file paths; the evidence pack intentionally omits them.
- Do not modify files, SQLite, playback state, or profile rules.
- Distinguish observed preferences from weak hypotheses in `confidence` and `rationale`.
- If evidence is insufficient, return an empty `suggestions` array.

## Supported Rule Types

- `prefer_lower_loudness`
- `prefer_lower_energy`
- `prefer_lower_vocalness`
- `demote_high_bpm`
- `prefer_artist`

Use `scope` values such as `global`, `focus`, `calm`, `decompress`, `stabilize`, or `energized`.

## Output Example

```json
{
  "suggestions": [
    {
      "suggestion_id": "focus-lower-loudness",
      "scope": "focus",
      "rule_type": "prefer_lower_loudness",
      "target": "loudness",
      "threshold": -12.0,
      "weight": 0.7,
      "confidence": "medium",
      "rationale": "The user repeatedly marked focus-session tracks as too loud.",
      "evidence_count": 2
    }
  ]
}
```
