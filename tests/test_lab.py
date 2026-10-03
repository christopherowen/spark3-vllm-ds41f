"""Host-independent tests for the experiment window runner (scripts/lab.py)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("lab", str(ROOT / "scripts" / "lab.py"))
spec = importlib.util.spec_from_loader("lab", loader)
lab = importlib.util.module_from_spec(spec)
loader.exec_module(lab)
# Host-independent: read the shipped example, not the site's git-ignored config/nodes.json.
lab.spark3.site_nodes = lambda: lab.spark3.read_json("config/nodes.example.json")

SPEC = {
    "experiment": "experiments/2026-09-29-determinism",
    "run": "lab0",
    "jobs": [
        {"kind": "measure", "profile": "lean", "bracket": True,
         "arms": [{"config": "cluster-r5o-pin.json", "label": "r5o"},
                  {"config": "cluster-detm-r5o-ref4d-b4144-pin.json", "label": "ref4d-b4144"}]},
        {"kind": "validate", "config": "cluster-detm-r5o-ref4d-b4144-trace8.json", "name": "ref4d-b4144",
         "chunk": 4096, "tier": "quick"},
        {"kind": "kernel", "bundles": [{"bundle": "b/mhc", "node": "dgx2"}]},
    ],
}


class HoldTest(unittest.TestCase):
    def test_new_hold_is_ours_and_carries_its_cap(self) -> None:
        hold = lab.new_hold(30, "run lab0", at=1_000_000.0)
        self.assertTrue(lab.hold_is_ours(hold))
        self.assertEqual(lab.parse_time(hold["expected_end"]) - lab.parse_time(hold["since"]), 1800)
        self.assertAlmostEqual(lab.heartbeat_age(hold, at=1_000_090.0), 90, delta=1)

    def test_foreign_or_missing_holds_are_not_ours(self) -> None:
        self.assertFalse(lab.hold_is_ours(None))
        self.assertFalse(lab.hold_is_ours({"holder": "codex tool-eval-bench"}))

    def test_window_closes_on_request_or_cap(self) -> None:
        hold = lab.new_hold(10, "", at=1_000_000.0)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "REQUEST", Path(directory) / "request.json"):
            self.assertIsNone(lab.window_should_close(hold, at=1_000_060.0))
            self.assertEqual(lab.window_should_close(hold, at=1_000_000.0 + 601), "time cap reached")
            lab.REQUEST.write_text("{}")
            self.assertIn("requested", lab.window_should_close(hold, at=1_000_060.0))

    def test_read_hold_tolerates_damage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hold.json"
            self.assertIsNone(lab.read_hold(path))
            path.write_text("{not json")
            self.assertFalse(lab.hold_is_ours(lab.read_hold(path)))


class PlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = lab.plan(SPEC)

    def test_measure_brackets_the_first_arm_and_never_stops_between_arms(self) -> None:
        boots = [s for s in self.steps if s["kind"] == "boot"]
        self.assertEqual([b["label"] for b in boots[:3]], ["r5o", "ref4d-b4144", "r5o-end"])
        first_validate = next(i for i, s in enumerate(self.steps) if s.get("label") == "ref4d-b4144" and
                              s["kind"] == "boot" and "trace8" in s["config"])
        self.assertNotIn("stop", [s["kind"] for s in self.steps[:first_validate]])

    def test_measure_ends_with_the_comparison_table(self) -> None:
        table = next(s for s in self.steps if s["kind"] == "table")
        self.assertEqual(table["argv"][1:], ["lab0", "r5o", "ref4d-b4144", "r5o-end"])

    def test_lean_profile_and_result_layout(self) -> None:
        bench = next(s for s in self.steps if s["kind"] == "cli" and s["label"] == "ref4d-b4144")
        self.assertIn("1024,16384", bench["argv"])
        self.assertEqual(bench["argv"][-1], "results/private/bench/lab0-ref4d-b4144")
        outputs = {s["out"] for s in self.steps if s["kind"] == "script" and s.get("label") == "r5o"}
        self.assertIn("results/private/determinism/lab0/c1-distinct-r5o.jsonl", outputs)
        self.assertIn("results/private/determinism/lab0/mixed-r5o.jsonl", outputs)
        c1 = next(s for s in self.steps if s["kind"] == "script" and "c1-distinct-r5o" in s["out"])
        self.assertEqual(c1["argv"][-2:], ["--prompts", "12"])

    def test_quick_validation_restarts_without_mixes_then_analyses_stopped(self) -> None:
        validate = self.steps[next(i for i, s in enumerate(self.steps) if s["kind"] == "fresh_inventory") - 1:]
        kinds = [s["kind"] for s in validate]
        self.assertEqual(kinds.count("boot"), 2)
        scripts = [" ".join(s["argv"]) for s in validate if s["kind"] == "script"]
        self.assertFalse(any("trace_mixes" in s for s in scripts))
        self.assertTrue(any("--suffix -bootB" in s and "--scenarios cache_long,distinct" in s for s in scripts))
        self.assertTrue(all("--chunk 4096" in s for s in scripts if "scenario_trace" in s))
        analyze_at = kinds.index("analyze")
        self.assertEqual(kinds[analyze_at - 1], "stop")
        self.assertEqual(validate[analyze_at]["dirs"], ["c8", "scenarios"])

    def test_full_validation_keeps_mixes_and_every_scenario(self) -> None:
        full = dict(SPEC, jobs=[dict(SPEC["jobs"][1], tier="full")])
        scripts = [" ".join(s["argv"]) for s in lab.plan(full) if s["kind"] == "script"]
        self.assertTrue(any("trace_mixes" in s for s in scripts))
        self.assertFalse(any("--scenarios" in s for s in scripts))

    def test_kernel_jobs_run_with_the_cluster_stopped(self) -> None:
        kernel_at = [s["kind"] for s in self.steps].index("kernel")
        self.assertEqual(self.steps[kernel_at - 1]["kind"], "stop")

    def test_unknown_job_kind_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            lab.plan(dict(SPEC, jobs=[{"kind": "mystery"}]))


class BootTimeTest(unittest.TestCase):
    def test_boot_rejects_a_different_physical_topology_before_launch(self) -> None:
        from unittest import mock

        promoted = {"nodes": [{"rank": 0}, {"rank": 1}, {"rank": 2}]}
        candidate = {"nodes": [*promoted["nodes"], {"rank": 3}]}
        with mock.patch.object(lab.spark3, "configuration", return_value=({}, candidate, {})), \
             mock.patch.object(lab, "nodes_config", return_value=promoted), \
             mock.patch.object(lab, "spark3_cli") as launch:
            self.assertFalse(lab.boot("ring4.json"))
            launch.assert_not_called()

    def test_ready_seconds_come_from_the_cluster_ready_line(self) -> None:
        lines = ["dgx1: steady memguard active", "cluster ready; memory guards are active on all nodes (+116.9s)"]
        self.assertEqual(lab.ready_seconds(lines), 116.9)
        self.assertIsNone(lab.ready_seconds(["boot failed"]))


class ProfilePlanTest(unittest.TestCase):
    def test_profile_job_traces_each_workload_then_compares_stopped(self) -> None:
        spec = {"experiment": "e", "run": "costs3", "jobs": [{
            "kind": "profile", "workloads": ["decode", "prefill"],
            "arms": [{"config": "cluster-r5o-pin-prof.json", "label": "r5o"},
                     {"config": "cluster-cand-prof.json", "label": "cand"}]}]}
        steps = lab.plan(spec)
        kinds = [s["kind"] for s in steps]
        self.assertEqual(kinds, ["boot", "curves", "profile", "profile", "boot", "curves", "profile", "profile", "stop", "costs"])
        self.assertEqual(steps[3]["argv"][-2:], ["--tokens", "16384"])
        self.assertEqual(steps[1]["out"], "results/private/determinism/costs3/curves-r5o.json")
        self.assertEqual(steps[-1]["labels"], ["r5o", "cand"])
        self.assertEqual(steps[-1]["out"], "results/private/determinism/costs3")


class QueueTest(unittest.TestCase):
    def test_add_validates_and_orders_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "QUEUE", Path(directory) / "queue"):
            good = Path(directory) / "a.json"
            good.write_text(json.dumps(SPEC))
            lab.queue_add(str(good))
            bad = Path(directory) / "b.json"
            bad.write_text(json.dumps(dict(SPEC, jobs=[{"kind": "mystery"}])))
            with self.assertRaises(SystemExit):
                lab.queue_add(str(bad))
            self.assertEqual([p.name.split("-", 1)[1] for p in lab.queued()], ["a.json"])

    def test_sync_is_a_job(self) -> None:
        steps = lab.plan(dict(SPEC, jobs=[{"kind": "sync"}]))
        self.assertEqual([s["kind"] for s in steps], ["sync"])


class CustomWorkloadTest(unittest.TestCase):
    def test_a_job_may_skip_the_bench_and_name_its_own_workloads(self) -> None:
        spec = {"experiment": "experiments/x", "run": "s1", "jobs": [{
            "kind": "measure", "bench": False,
            "extras": [["hol_latency.py", ["--rounds", "2"], "hol"]],
            "arms": [{"config": "cluster-a.json", "label": "a"}]}]}
        steps = lab.plan(spec)
        self.assertEqual([s["kind"] for s in steps], ["boot", "curves", "script", "table"])
        self.assertEqual(steps[2]["out"], "results/private/determinism/s1/hol-a.jsonl")
        self.assertEqual(steps[3]["summaries"], ["results/private/determinism/s1/hol-a.jsonl"])


class CurvesTest(unittest.TestCase):
    def test_curve_table_compares_arms_at_fixed_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            base = Path(directory) / "curves-r5o.json"
            base.write_text(json.dumps({"verify": [[1, 10.0], [6, 12.0], [48, 20.0]], "draft": [[1, 2.0]]}))
            cand = Path(directory) / "curves-cand.json"
            cand.write_text(json.dumps({"verify": [[1, 11.0], [6, 12.0], [48, 19.0]], "draft": [[1, 2.0]]}))
            table = lab.curves_table(["curves-r5o.json", "curves-cand.json", "curves-missing.json"])
        self.assertIn("1:11.00 (+10.0%)", table)
        self.assertIn("48:19.00 (-5.0%)", table)
        self.assertEqual(table.count("\n"), 2)

    def test_padded_and_real_row_curves_never_compare(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            (Path(directory) / "curves-a.json").write_text(json.dumps({"verify": [[1, 10.0]], "draft": []}))
            (Path(directory) / "curves-b.json").write_text(
                json.dumps({"rows": "real", "verify": [[1, 15.0]], "draft": []}))
            (Path(directory) / "curves-c.json").write_text(
                json.dumps({"rows": "real", "verify": [[1, 16.5]], "draft": []}))
            table = lab.curves_table(["curves-a.json", "curves-b.json", "curves-c.json"])
        self.assertIn("b              real   1:15.00", table)
        self.assertNotIn("+50.0%", table)
        self.assertIn("1:16.50 (+10.0%)", table)


class OverlayTest(unittest.TestCase):
    def test_fewer_fences_than_the_image_is_a_problem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "stale.py"
            stale.write_text("x = 1\n")
            fresh = Path(directory) / "fresh.py"
            fresh.write_text("cute.arch.fence_proxy('async.shared')\n")
            image = Path(directory) / "image.py"
            image.write_text("cute.arch.fence_proxy('async.shared')\n")
            mounts = [[str(stale), "/opt/spark3/candidate/b12x/b12x/gemm/a.py", "ro"],
                      [str(fresh), "/opt/spark3/candidate/b12x/b12x/gemm/b.py", "ro"],
                      [str(stale), "/opt/spark3/candidate/vllm/vllm/x.py", "ro"]]
            problems = lab.fence_problems(mounts, "img", "/home/x", read_image=lambda path: image)
        self.assertEqual(len(problems), 1)
        self.assertIn("stale.py", problems[0])


class KernelLabTest(unittest.TestCase):
    CANDIDATE = {"image": "img:tag", "env": {"B12X_AUTOTUNE": "0"}, "argv": ["/r/replay.py", "/cap"],
                 "mounts": [["~/captures", "/cap"]]}

    def test_overlay_files_mount_at_the_candidate_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            target = bundle / "overlay" / "b12x" / "norm" / "mhc" / "_kernels.py"
            target.parent.mkdir(parents=True)
            target.write_text("")
            vllm = bundle / "overlay" / "vllm" / "models" / "deepseek_v4_1" / "b12x_layers.py"
            vllm.parent.mkdir(parents=True)
            vllm.write_text("")
            command = lab.bundle_command(bundle, self.CANDIDATE, bundle / "out")
        joined = " ".join(command)
        self.assertIn(":/opt/spark3/candidate/b12x/b12x/norm/mhc/_kernels.py:ro", joined)
        self.assertIn(":/opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/b12x_layers.py:ro", joined)
        self.assertIn("B12X_AUTOTUNE=0", command)
        self.assertEqual(command[-3:], ["img:tag", "/r/replay.py", "/cap"])

    def test_verdict_passes_only_invariant_and_bit_equal(self) -> None:
        good = [json.dumps({"op": "mhc production layer 1 pre", "groups": 2}),
                json.dumps({"op": "mhc candidate layer 1 pre", "groups": 1}),
                json.dumps({"bits": "tf32x-a vs tf32-s40 layer 1 pre", "equal_at": [1, 2], "differs_at": []}),
                json.dumps({"timing_us": "candidate layer 1 pre", "1": 10.0})]
        self.assertTrue(lab.verdict_from_lines(good, self.CANDIDATE)["passed"])
        variant = good + [json.dumps({"op": "mhc tf32x-b layer 1 pre", "groups": 2})]
        self.assertEqual(lab.verdict_from_lines(variant, self.CANDIDATE)["batch_variant"],
                         ["mhc tf32x-b layer 1 pre"])
        unequal = good + [json.dumps({"bits": "x vs y", "equal_at": [1], "differs_at": [2]})]
        self.assertFalse(lab.verdict_from_lines(unequal, self.CANDIDATE)["passed"])
        crashed = good + ["Traceback (most recent call last):"]
        self.assertFalse(lab.verdict_from_lines(crashed, self.CANDIDATE)["passed"])
        self.assertFalse(lab.verdict_from_lines([], self.CANDIDATE)["passed"])

    def test_suite_bundles_pass_on_passing_tests_only(self) -> None:
        suite = dict(self.CANDIDATE, verdict="exit")
        good = ["tests/kernels/moe/test_x.py ....", "===== 18 passed, 1 warning in 40.2s ====="]
        verdict = lab.verdict_from_lines(good, suite)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["summary"], "18 passed, 1 warning in 40.2s")
        failed = ["FAILED tests/kernels/moe/test_x.py::test_a[1] - AssertionError: rows differ",
                  "===== 1 failed, 17 passed in 41.0s ====="]
        verdict = lab.verdict_from_lines(failed, suite)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["failed"], ["tests/kernels/moe/test_x.py::test_a[1]"])
        for lines in ([], ["===== 3 skipped in 1.0s ====="], ["===== 2 passed, 1 error in 3.0s ====="]):
            with self.subTest(lines=lines):
                self.assertFalse(lab.verdict_from_lines(lines, suite)["passed"])
        a = dict(lab.verdict_from_lines(failed, suite), bundle="b")
        self.assertFalse(lab.verdicts_agree([a, dict(lab.verdict_from_lines(good, suite), bundle="b")]))
        self.assertTrue(lab.verdicts_agree([a, dict(a, node="dgx4")]))

    def test_nodes_must_agree_on_groups_and_bits(self) -> None:
        a = {"bundle": "b", "groups": [{"op": "x", "groups": 1}], "bits": [{"bits": "p", "differs_at": []}]}
        b = dict(a, node="dgx2")
        self.assertTrue(lab.verdicts_agree([a, b]))
        c = dict(a, groups=[{"op": "x", "groups": 2}])
        self.assertFalse(lab.verdicts_agree([a, c]))


class AnalysisTest(unittest.TestCase):
    def test_summary_counts_rows_and_cross_boot_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            records = [
                {"request": "a-r0", "against": "a-r0-bootB", "rows_compared": 10, "rows_differing": 0,
                 "outputs_equal": True},
                {"request": "a-r0", "against": "a-r1", "rows_compared": 5, "rows_differing": 0,
                 "outputs_equal": True},
                {"request": "a-r0", "rows": 12, "recomputed_mismatch": []},
            ]
            (out / "analysis4-scenarios-dgx1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            summary = lab.summarize_analysis(out)
            self.assertTrue(summary["passed"])
            entry = summary["analysis4-scenarios-dgx1"]
            self.assertEqual((entry["pairs"], entry["rows"], entry["cross_boot_pairs"]), (2, 15, 1))
            records[1]["rows_differing"] = 3
            (out / "analysis4-scenarios-dgx1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            self.assertFalse(lab.summarize_analysis(out)["passed"])

    def test_no_analysis_files_is_not_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(lab.summarize_analysis(Path(directory))["passed"])


if __name__ == "__main__":
    unittest.main()
