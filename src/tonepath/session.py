"""Stateful session runtime for Tonepath."""

from __future__ import annotations

from dataclasses import replace

from tonepath.db import TonepathStore
from tonepath.explanation import explain_candidate
from tonepath.models import CandidateScore, FeedbackType, SessionPhase, SessionPlan
from tonepath.planner import plan_session
from tonepath.selector import select_path


class SessionRunner:
    """Run a mutable listening session over the deterministic selector."""

    def __init__(
        self,
        store: TonepathStore,
        prompt: str,
        limit_per_phase: int = 2,
        plan: SessionPlan | None = None,
    ) -> None:
        self.store = store
        self.prompt = prompt
        self.limit_per_phase = limit_per_phase
        self.base_plan = plan or plan_session(prompt)
        self.session_id = store.save_session(self.base_plan)
        self.current_index = 0
        self.energy_delta = 0.0
        self.force_no_vocals = self.base_plan.request.no_vocals
        self.queue = select_path(store, self.active_plan(), limit_per_phase=limit_per_phase)

    def active_plan(self) -> SessionPlan:
        """Return the current plan after session-level feedback adjustments."""

        phases = tuple(self.adjust_phase(phase) for phase in self.base_plan.phases)
        request = replace(self.base_plan.request, no_vocals=self.force_no_vocals)
        return SessionPlan(request=request, phases=phases)

    def current(self) -> CandidateScore | None:
        """Return the current candidate if one exists."""

        if self.current_index >= len(self.queue):
            return None
        return self.queue[self.current_index]

    def upcoming(self, limit: int = 6) -> list[CandidateScore]:
        """Return upcoming candidates after the current track."""

        start = min(self.current_index + 1, len(self.queue))
        return self.queue[start : start + limit]

    def move_next(self) -> bool:
        """Move to the next candidate without recording feedback."""

        if self.current_index + 1 >= len(self.queue):
            return False
        self.current_index += 1
        return True

    def move_previous(self) -> bool:
        """Move to the previous candidate without recording feedback."""

        if self.current_index <= 0:
            return False
        self.current_index -= 1
        return True

    def move_to_start(self) -> bool:
        """Move to the first candidate without recording feedback."""

        if not self.queue:
            return False
        self.current_index = 0
        return True

    def current_explanation(self) -> str:
        """Return an auditable explanation for the current track."""

        candidate = self.current()
        if candidate is None:
            return "No current track. Run `tonepath scan` first."
        return explain_candidate(self.store, candidate)

    def apply_feedback(self, feedback_type: FeedbackType) -> str:
        """Record feedback and update subsequent recommendations."""

        candidate = self.current()
        track_id = candidate.track.id if candidate and candidate.track.id is not None else None
        self.store.record_feedback(feedback_type, session_id=self.session_id, track_id=track_id)

        if feedback_type == "skip":
            if self.current_index < len(self.queue):
                self.current_index += 1
            self.rebuild_future()
            return "Skipped current track."
        if feedback_type == "no-vocals":
            self.force_no_vocals = True
            self.rebuild_future()
            return "No-vocals constraint applied to upcoming tracks."
        if feedback_type == "too-loud":
            self.energy_delta = max(self.energy_delta - 0.15, -0.35)
            self.rebuild_future()
            return "Reduced upcoming energy target."
        if feedback_type == "too-slow":
            self.energy_delta = min(self.energy_delta + 0.1, 0.3)
            self.rebuild_future()
            return "Raised upcoming energy target."
        if feedback_type == "like":
            self.rebuild_future()
            return "Stored like feedback."
        raise ValueError(f"Unsupported feedback type: {feedback_type}")

    def rebuild_future(self) -> None:
        """Re-select upcoming tracks while keeping already reached tracks stable."""

        keep = self.queue[: min(self.current_index + 1, len(self.queue))]
        used_ids = {candidate.track.id for candidate in keep if candidate.track.id is not None}
        future = select_path(
            self.store,
            self.active_plan(),
            limit_per_phase=self.limit_per_phase,
            excluded_track_ids=used_ids,
        )
        self.queue = [*keep, *future]

    def adjust_phase(self, phase: SessionPhase) -> SessionPhase:
        """Apply session feedback constraints to one phase."""

        energy = min(max(phase.target_energy + self.energy_delta, 0.0), 1.0)
        vocal_policy = "avoid" if self.force_no_vocals else phase.vocal_policy
        return replace(phase, target_energy=energy, vocal_policy=vocal_policy)
