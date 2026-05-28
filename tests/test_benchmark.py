import json
import unittest
from pathlib import Path

from tonepath.benchmark import (
    evaluate_benchmark_scenario,
    load_benchmark_scenarios,
    max_affect_axis_phase,
    metadata_hygiene_warning,
    min_valence_phase,
    no_duplicate_candidates,
    no_high_vocalness_top_k,
    required_affect_top_k,
    required_confidence_top_k,
)


class BenchmarkTest(unittest.TestCase):
    def test_packaged_benchmark_fixture_matches_test_fixture(self) -> None:
        packaged = Path("src/tonepath/resources/selection_benchmark.jsonl").read_text(encoding="utf-8")
        fixture = Path("tests/fixtures/selection_benchmark.jsonl").read_text(encoding="utf-8")

        self.assertEqual(packaged, fixture)

    def test_load_benchmark_scenarios_validates_fixture(self) -> None:
        scenarios = load_benchmark_scenarios()

        self.assertGreaterEqual(len(scenarios), 10)
        self.assertEqual(scenarios[0]["id"], "irritated_to_focus_no_vocals_zh")
        self.assertIn("expected_intent", scenarios[0])
        self.assertIsInstance(scenarios[0]["checks"], list)

    def test_fixture_does_not_use_track_allowlists_or_blocklists(self) -> None:
        for line in Path("tests/fixtures/selection_benchmark.jsonl").read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            encoded = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("allowlist", encoded)
            self.assertNotIn("blocklist", encoded)
            self.assertNotIn("track_title", encoded)

    def test_high_vocalness_check_fails_for_top_candidate(self) -> None:
        result = no_high_vocalness_top_k(
            {"type": "no_high_vocalness_top_k", "k": 3, "max_vocalness": 0.65, "level": "fail"},
            [candidate(0.4, -12.0, 100.0, 0.8)],
        )

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["affected_ranks"], [1])

    def test_required_confidence_check_catches_low_evidence(self) -> None:
        result = required_confidence_top_k(
            {
                "type": "required_confidence_top_k",
                "k": 2,
                "min_confidence": "medium",
                "require_feature_source": True,
                "level": "fail",
            },
            [candidate(0.4, -12.0, 100.0, 0.2, confidence="low", source=None)],
        )

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["affected_ranks"], [1])

    def test_required_affect_check_catches_missing_affect(self) -> None:
        result = required_affect_top_k({"type": "required_affect_top_k", "k": 1, "level": "warn"}, [candidate(0.4, -12.0, 100.0, 0.2)])

        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["affected_ranks"], [1])

    def test_valence_phase_check_catches_low_lift_valence(self) -> None:
        row = candidate(0.4, -12.0, 100.0, 0.2, valence=0.4, affect={"uplift": 0.2})
        row["phase"] = "lift"

        result = min_valence_phase({"type": "min_valence_phase", "phase": "lift", "min_valence": 0.5, "level": "warn"}, [row])

        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["affected_ranks"], [1])

    def test_affect_axis_phase_check_catches_dark_lift(self) -> None:
        row = candidate(0.4, -12.0, 100.0, 0.2, valence=0.6, affect={"darkness": 0.7})
        row["phase"] = "lift"

        result = max_affect_axis_phase(
            {"type": "max_affect_axis_phase", "phase": "lift", "axis": "darkness", "max_value": 0.6, "level": "warn"},
            [row],
        )

        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["affected_ranks"], [1])

    def test_duplicate_candidate_check_reports_later_duplicate(self) -> None:
        rows = [
            candidate(0.4, -12.0, 100.0, 0.2, key=["same", "artist", 18]),
            candidate(0.5, -13.0, 105.0, 0.3, key=["same", "artist", 18]),
        ]

        result = no_duplicate_candidates({"type": "no_duplicate_candidates", "level": "fail"}, rows)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["affected_ranks"], [2])

    def test_metadata_hygiene_warning_is_warn_only(self) -> None:
        row = candidate(0.4, -12.0, 100.0, 0.2)
        track = row["track"]
        self.assertIsInstance(track, dict)
        track["metadata_issues"] = ["dirty title"]

        result = metadata_hygiene_warning({"type": "metadata_hygiene_warning", "level": "warn"}, [row])

        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["affected_ranks"], [1])

    def test_benchmark_scenario_aggregates_failures(self) -> None:
        scenario = {
            "id": "unit",
            "lang": "en",
            "prompt": "focus",
            "limit": 1,
            "expected_intent": {"target_state": "focus", "constraints": ["avoid_vocals"]},
            "checks": [{"type": "no_high_vocalness_top_k", "k": 1, "max_vocalness": 0.65, "level": "fail"}],
        }

        result = evaluate_benchmark_scenario(
            scenario,
            {"target_state": "focus", "constraints": ["avoid_vocals"]},
            [candidate(0.4, -12.0, 100.0, 0.9)],
        )

        self.assertEqual(result["result"], "FAIL")
        self.assertEqual(result["checks"][0]["status"], "pass")
        self.assertEqual(result["checks"][1]["status"], "fail")


def candidate(
    energy: float,
    loudness: float,
    bpm: float,
    vocalness: float,
    confidence: str = "high",
    source: str | None = "test",
    key: list[object] | None = None,
    arousal: float | None = None,
    valence: float | None = None,
    affect: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return one benchmark candidate row."""

    return {
        "phase": "focus",
        "track": {
            "id": 1,
            "display_label": "track - artist",
            "canonical_key": key or ["track", "artist", 18],
            "metadata_issues": [],
        },
        "confidence": confidence,
        "features": {
            "source": source,
            "energy": energy,
            "loudness": loudness,
            "bpm": bpm,
            "vocalness": vocalness,
            "arousal": arousal,
            "valence": valence,
            "affect_profile": affect or {},
        },
    }


if __name__ == "__main__":
    unittest.main()
