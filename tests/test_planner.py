import unittest

from tonepath.planner import parse_request, plan_session


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


if __name__ == "__main__":
    unittest.main()

