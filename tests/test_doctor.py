"""Host-independent tests for doctor's node checks."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)


class DesktopTest(unittest.TestCase):
    def test_headless_node_passes(self) -> None:
        self.assertEqual(spark3.desktop_problems("dgx1", "multi-user.target\ninactive\n"), [])

    def test_graphical_target_and_running_display_manager_are_reported(self) -> None:
        problems = spark3.desktop_problems("dgx2", "graphical.target\nactive\n")
        self.assertEqual(len(problems), 2)
        self.assertIn("boots to graphical.target", problems[0])
        self.assertIn("display manager is active", problems[1])

    def test_display_manager_started_by_hand_is_reported(self) -> None:
        problems = spark3.desktop_problems("dgx3", "multi-user.target\nactive\n")
        self.assertEqual(problems, ["dgx3: display manager is active"])

    def test_missing_output_is_reported(self) -> None:
        problems = spark3.desktop_problems("dgx1", "")
        self.assertEqual(len(problems), 1)
        self.assertIn("unknown target", problems[0])


if __name__ == "__main__":
    unittest.main()
