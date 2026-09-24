"""Host-independent tests for the build directory identity."""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)


class BuildDirectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = spark3.build_inputs(spark3.read_json("upstreams.lock.json"))

    def test_inputs_come_from_the_lock_and_source_manifest(self) -> None:
        upstreams = spark3.read_json("upstreams.lock.json")
        manifest = spark3.read_json(upstreams["source_manifest"])
        for name in spark3.BUILD_PROJECTS:
            self.assertEqual(self.inputs[name]["revision"], upstreams["sources"][name]["revision"])
            self.assertEqual(self.inputs[name]["expected_tree"], manifest[name]["expected_tree"])
            self.assertEqual(self.inputs[name]["patch_head"], manifest[name]["patch_head"])

    def test_directory_is_stable_and_follows_every_input(self) -> None:
        directory = spark3.build_directory(self.inputs)
        self.assertEqual(directory, spark3.build_directory(copy.deepcopy(self.inputs)))
        self.assertEqual(directory.parent, ROOT / ".work/build")
        self.assertTrue(directory.name.startswith(
            f"vllm-{self.inputs['vllm']['revision'][:12]}-b12x-{self.inputs['b12x']['revision'][:12]}-"
        ))
        changed = copy.deepcopy(self.inputs)
        changed["cutlass"]["revision"] = "0" * 40
        self.assertNotEqual(directory, spark3.build_directory(changed))

    def test_tree_digest_ignores_git_metadata(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.py").write_text("x = 1\n")
            before = spark3.tree_digest(root)
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
            self.assertEqual(before, spark3.tree_digest(root))
            (root / "a.py").write_text("x = 2\n")
            self.assertNotEqual(before, spark3.tree_digest(root))

    def test_normalize_modes_uses_git_modes(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "d").mkdir(mode=0o775)
            plain = root / "d" / "plain.txt"
            plain.write_text("x")
            plain.chmod(0o664)
            tool = root / "tool.sh"
            tool.write_text("#!/bin/sh\n")
            tool.chmod(0o775)
            spark3.normalize_modes(root)
            self.assertEqual(plain.stat().st_mode & 0o777, 0o644)
            self.assertEqual(tool.stat().st_mode & 0o777, 0o755)
            self.assertEqual((root / "d").stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
