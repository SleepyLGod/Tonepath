import tempfile
import unittest
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.models import SessionPlan, SessionRequest
from tonepath.planner import build_phases
from tonepath.privacy import delete_profile, privacy_status


class PrivacyTest(unittest.TestCase):
    def test_delete_profile_removes_feedback_and_sessions_but_keeps_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            request = SessionRequest("focus", "irritated", "focus", 1800)
            session_id = store.save_session(SessionPlan(request, tuple(build_phases(request))))
            store.record_feedback("like", session_id=session_id)
            before = store.profile_summary()
            self.assertEqual(before["sessions"], 1)
            self.assertEqual(before["feedback"], 1)
            delete_profile(store)
            after = store.profile_summary()
            self.assertEqual(after["sessions"], 0)
            self.assertEqual(after["feedback"], 0)
            self.assertEqual(after["profile_rules"], 0)
            store.close()

    def test_privacy_status_says_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            status = privacy_status(store)
            self.assertIn("offline by default", status)
            self.assertIn("Sent to LLM: none", status)
            store.close()


if __name__ == "__main__":
    unittest.main()
