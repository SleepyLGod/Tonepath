# Tonepath Algorithm Backlog

This backlog captures recommendation-quality issues that should be evaluated separately from memory/profile UX work. Do not use this file as a selector tuning checklist without first reproducing the issue with `eval diagnose`, `eval suite`, or real listening notes.

## Open Questions

- **Intro vs full-track feel:** some tracks start brightly but settle into a calmer overall mood. Current features are mostly whole-track summaries and do not model intro/body changes.
- **Rhythm vs emotional color:** some songs have usable rhythm but feel gloomy or tense. Selector evidence needs better separation between movement, valence, tension, and darkness.
- **Sad to gently uplifting:** the path should lift gradually without jumping to loud, vocal-heavy, or dramatic tracks.
- **CLAP / hybrid promotion gate:** CLAP and hybrid remain evaluation-only until they show stable improvement over selector across emotion-transition scenarios.
- **Library coverage:** if a target mood has too few good candidates, selector tuning cannot create suitable music. Benchmark failures should distinguish library gaps from scoring mistakes.

## Current Rule

Keep algorithm and model changes out of Phase 2 memory/profile implementation unless a failing validation run clearly points to selector tuning or weak model evidence.
