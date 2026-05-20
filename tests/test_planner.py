import json
import unittest
from pathlib import Path

from tonepath.planner import parse_request, plan_session, request_constraints


class PlannerTest(unittest.TestCase):
    def test_chinese_focus_prompt(self) -> None:
        request = parse_request("我现在很烦，想半小时后进入写代码状态，不要人声")
        self.assertEqual(request.source_state, "irritated")
        self.assertEqual(request.target_state, "focus")
        self.assertEqual(request.duration_sec, 30 * 60)
        self.assertTrue(request.no_vocals)

    def test_plan_has_ordered_phases(self) -> None:
        plan = plan_session("从烦躁到专注，30分钟")
        self.assertGreaterEqual(len(plan.phases), 3)
        self.assertEqual(plan.phases[0].start_sec, 0)
        self.assertEqual(plan.phases[-1].end_sec, 30 * 60)

    def test_bilingual_intent_fixture_corpus(self) -> None:
        for case in intent_cases():
            with self.subTest(prompt=case["prompt"]):
                request = parse_request(str(case["prompt"]))
                self.assertEqual(request.source_state, case["source_state"])
                self.assertEqual(request.target_state, case["target_state"])
                self.assertEqual(request.duration_sec // 60, case["duration_min"])
                self.assertEqual(request.no_vocals, case["no_vocals"])
                self.assertEqual(request.quiet, case["quiet"])

    def test_packaged_intent_corpus_matches_test_fixture(self) -> None:
        packaged = Path(__file__).parents[1] / "src" / "tonepath" / "resources" / "intent_prompts.jsonl"
        fixture = Path(__file__).parent / "fixtures" / "intent_prompts.jsonl"

        self.assertEqual(packaged.read_text(encoding="utf-8"), fixture.read_text(encoding="utf-8"))

    def test_request_constraints_include_low_stimulation(self) -> None:
        request = parse_request("我要写论文，四十五分钟，低刺激，最好不要人声")

        self.assertEqual(request_constraints(request), ["avoid_vocals", "low_stimulation"])

    def test_quiet_focus_keeps_phase_labels_but_lowers_targets(self) -> None:
        normal = plan_session("我要写论文，四十五分钟").phases
        quiet = plan_session("我要写论文，四十五分钟，低刺激").phases

        self.assertEqual([phase.label for phase in quiet], ["decompress", "stabilize", "focus"])
        self.assertLess(quiet[-1].target_energy, normal[-1].target_energy)
        self.assertLess(quiet[-1].target_arousal, normal[-1].target_arousal)

    def test_quiet_calm_keeps_phase_labels_but_lowers_targets(self) -> None:
        normal = plan_session("晚上准备睡觉，二十分钟").phases
        quiet = plan_session("晚上准备睡觉，二十分钟，很安静").phases

        self.assertEqual([phase.label for phase in quiet], ["soften", "settle", "calm"])
        self.assertLess(quiet[-1].target_energy, normal[-1].target_energy)
        self.assertLess(quiet[-1].target_arousal, normal[-1].target_arousal)


def intent_cases() -> list[dict[str, object]]:
    """Return the checked-in bilingual prompt-intent fixtures."""

    path = Path(__file__).parent / "fixtures" / "intent_prompts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
