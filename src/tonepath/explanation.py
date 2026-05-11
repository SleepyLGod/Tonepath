"""Auditable explanations for selected tracks."""

from __future__ import annotations

from tonepath.db import TonepathStore
from tonepath.models import CandidateScore


def explain_candidate(store: TonepathStore, candidate: CandidateScore) -> str:
    """Explain a selection using only stored metadata, features, phases, and feedback."""

    track = candidate.track
    features = store.get_features(track.id) if track.id is not None else None
    feedback = store.feedback_counts_for_track(track.id) if track.id is not None else {}
    lines = [
        "选择原因：",
        f"- 当前阶段：{candidate.phase.label}",
        f"- 目标 energy：{candidate.phase.target_energy:.2f}",
        f"- Confidence：{candidate.confidence}",
        f"- 曲目：{track.title or 'unknown'}",
        f"- 艺人：{track.artist or 'unknown'}",
        f"- Genre：{track.genre or 'unknown'}",
    ]
    if track.duration is not None:
        lines.append(f"- Duration：{track.duration:.0f}s")
    else:
        lines.append("- Duration：unknown")

    if features is None:
        lines.append("- BPM：unknown（未做本地音频分析）")
        lines.append("- Vocalness：unknown（未做本地音频分析）")
    else:
        if features.bpm is None:
            lines.append("- BPM：unknown")
        else:
            lines.append(f"- BPM：{features.bpm:.0f}")
        if features.vocalness is None:
            lines.append("- Vocalness：unknown")
        else:
            lines.append(f"- Vocalness：{features.vocalness:.2f}")

    if feedback:
        lines.append(f"- 本地反馈：{feedback}")
    else:
        lines.append("- 本地反馈：none")

    for reason in candidate.reasons:
        lines.append(f"- Scoring note：{reason}")
    return "\n".join(lines)

