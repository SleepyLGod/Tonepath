import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tonepath.preparation import PreparationOptions, ScanSummary, resolve_prepare_mode, run_preparation


class PreparationServiceTest(unittest.TestCase):
    def test_resolve_prepare_mode_rejects_conflicting_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            resolve_prepare_mode("balanced", fast=True, full=True)

    def test_balanced_prepare_runs_scan_mir_tags_and_affect(self) -> None:
        store = MagicMock()
        events: list[str] = []
        with patch("tonepath.preparation.TonepathStore", return_value=store), patch(
            "tonepath.preparation.scan_library",
            return_value=ScanSummary(total=2, scanned_dirs=1, skipped=0, pruned=0),
        ), patch(
            "tonepath.preparation.analyze_library",
            side_effect=[(2, 0), (2, 0), (2, 0)],
        ) as analyze, patch(
            "tonepath.preparation.model_runtime_status",
            return_value=SimpleNamespace(ready=True, affect_ready=True),
        ), patch(
            "tonepath.preparation.library_status",
            return_value=SimpleNamespace(tracks=2),
        ):
            result = run_preparation(
                PreparationOptions(paths=(Path("/music"),), mode="balanced", limit=5),
                on_event=lambda event: events.append(event.message),
            )

        self.assertEqual(analyze.call_count, 3)
        self.assertEqual(analyze.call_args_list[0].kwargs["features"], "mir")
        self.assertEqual(analyze.call_args_list[1].kwargs["features"], "tags")
        self.assertEqual(analyze.call_args_list[2].kwargs["features"], "affect")
        self.assertTrue(result.runtime_ready)
        self.assertIn("Prepare: scan", events)
        self.assertIn("Scanned 2 track(s) from 1 directory.", events)
        store.close.assert_called_once()

    def test_missing_models_still_runs_base_mir_without_download(self) -> None:
        store = MagicMock()
        with patch("tonepath.preparation.TonepathStore", return_value=store), patch(
            "tonepath.preparation.scan_library",
            return_value=ScanSummary(total=1, scanned_dirs=1, skipped=0, pruned=0),
        ), patch(
            "tonepath.preparation.analyze_library",
            return_value=(1, 0),
        ) as analyze, patch(
            "tonepath.preparation.model_runtime_status",
            return_value=SimpleNamespace(ready=False, affect_ready=False),
        ), patch("tonepath.preparation.setup_essentia_tf_runtime") as setup, patch(
            "tonepath.preparation.library_status",
            return_value=SimpleNamespace(tracks=1),
        ):
            result = run_preparation(PreparationOptions(paths=(Path("/music"),), mode="balanced"))

        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(analyze.call_args.kwargs["features"], "mir")
        setup.assert_not_called()
        self.assertFalse(result.runtime_ready)

    def test_model_setup_is_an_explicit_option(self) -> None:
        store = MagicMock()
        with patch("tonepath.preparation.TonepathStore", return_value=store), patch(
            "tonepath.preparation.scan_library",
            return_value=ScanSummary(total=1, scanned_dirs=1, skipped=0, pruned=0),
        ), patch(
            "tonepath.preparation.analyze_library",
            side_effect=[(1, 0), (1, 0), (1, 0)],
        ), patch(
            "tonepath.preparation.model_runtime_status",
            return_value=SimpleNamespace(ready=False, affect_ready=False),
        ), patch(
            "tonepath.preparation.setup_essentia_tf_runtime",
            return_value=SimpleNamespace(ready=True, affect_ready=True),
        ) as setup, patch(
            "tonepath.preparation.library_status",
            return_value=SimpleNamespace(tracks=1),
        ):
            run_preparation(
                PreparationOptions(paths=(Path("/music"),), mode="full", setup_models=True)
            )

        setup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
