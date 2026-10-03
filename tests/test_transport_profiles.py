"""Named profiles must preserve measured settings and reject silent drift."""

import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3_tuning_test", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
profiles = spark3.transport_profiles


class TransportProfilesTest(unittest.TestCase):
    def resolve(self, name):
        return profiles.resolve(ROOT, profiles.DEFAULT_PROFILES, name)

    def test_profiles_reproduce_base_execution_settings(self):
        for name, node_file in (("tp3", "config/nodes.example.json"),
                                ("tp4", "config/examples/nodes-ring4.json")):
            with self.subTest(name=name):
                profile, resolved = self.resolve(name)
                base = spark3.read_json(profile["base_config"])
                self.assertEqual(resolved, base)
                nodes = spark3.read_json(node_file)
                result = profiles.materialize(profile, resolved, nodes, node_file)
                self.assertEqual(result["environment"], base["environment"])
                self.assertEqual(result["serve_args"], base["serve_args"])
                self.assertEqual(result["container"], base["container"])
                self.assertEqual(spark3.topology.problems(result, nodes), [])
                self.assertFalse(result["deployment"]["launch_enabled"])
                self.assertNotIn("branch", result["deployment"])
                self.assertEqual(resolved, base)

    def test_derived_padding_matches_audited_layouts(self):
        for name, expected in (("tp3", (72, 9, 25632, 129408, 768)),
                               ("tp4", (64, 8, 25600, 129280, 576))):
            with self.subTest(name=name):
                _, cluster = self.resolve(name)
                original = copy.deepcopy(cluster)
                report = spark3.model_layout.describe(ROOT, cluster, 2097152)
                dims = report["model_dimensions"]
                actual = tuple(dims[key]["padded_global"] for key in (
                    "attention_heads", "attention_output_groups", "engram_wkv_width", "target_vocabulary_rows"))
                actual += (dims["routed_expert_intermediate_width"]["allocated_per_rank"],)
                self.assertEqual(actual, expected)
                self.assertEqual(cluster, original)
                self.assertEqual(report["scheduled_rows"]["prefill_sp_min_live_rows"], 205)
                rows = {r["live_rows"]: r for r in report["scheduled_rows"]["examples"]}
                self.assertEqual(rows[19]["next_configured_graph_capacity"], 20)
                self.assertEqual(rows[19]["next_draft_context_graph_capacity"], 32)
                self.assertIsNone(rows[19]["prefill_sp_collective_rows_if_eligible"])
                self.assertEqual(rows[205]["prefill_sp_collective_rows_if_eligible"], 207 if name == "tp3" else 208)
                self.assertEqual(rows[4096]["prefill_sp_collective_rows_if_eligible"], 4098 if name == "tp3" else 4096)
                self.assertEqual(report["kernel_scratch"]["compact_moe_intermediate_width_when_selected"],
                                 None if name == "tp3" else 640)

    def test_draft_storage_and_graph_alignment(self):
        for name in ("tp3", "tp4"):
            _, cluster = self.resolve(name)
            report = spark3.model_layout.describe(ROOT, cluster, 2097152)
            draft = report["drafter"]
            tp3 = name == "tp3"
            self.assertEqual([r["real_rows"] for r in draft["vocabulary_shards"]],
                             [43136, 43136, 43008] if tp3 else [32320] * 4)
            self.assertEqual([r["padding_rows"] for r in draft["vocabulary_shards"]],
                             [0, 0, 128] if tp3 else [0] * 4)
            if tp3:
                # The promoted TP3 profile still re-quantizes both drafter heads.
                self.assertEqual(draft["head_formats"], {"lm_head": "nvfp4", "markov_output": "nvfp4"})
                self.assertEqual(draft["lm_head_nvfp4"]["packed_values_uint8_shape"], [43136, 2560])
                self.assertEqual(draft["lm_head_nvfp4"]["swizzled_scales_e4m3_storage_shape"], [43136, 320])
                self.assertEqual(draft["lm_head_nvfp4"]["scale_alignment_extra_bytes"], 0)
                self.assertEqual(draft["markov_output_nvfp4"]["scale_alignment_extra_bytes"], 0)
                self.assertEqual(len(draft["quantized_activation_examples"]), 3)
            else:
                # The TP4 candidate keeps the checkpoint's BF16 heads; the draft
                # head is the target head's own tensor.
                self.assertEqual(draft["head_formats"], {"lm_head": "bf16", "markov_output": "bf16"})
                self.assertEqual(draft["lm_head_bf16"],
                                 {"logical_shape": [32320, 5120], "bf16_bytes": 330956800,
                                  "shared_with_target_head": True})
                self.assertEqual(draft["markov_output_bf16"],
                                 {"logical_shape": [32320, 256], "bf16_bytes": 16547840})
                self.assertNotIn("lm_head_nvfp4", draft)
                self.assertEqual(draft["quantized_activation_examples"], [])
            self.assertEqual(draft["aux_context_bf16_buffer_shape"], [48, 15360])
            self.assertEqual(report["model_dimensions"]["draft_aux_projection_output"]["allocated_per_rank"],
                             1728 if tp3 else 1280)
            graphs = report["scheduled_rows"]
            self.assertEqual(graphs["draft_context_graph_capacities_if_full_supported"], [1, 2, 4, 8, 16, 32, 48])
            self.assertEqual(graphs["draft_query_graph_capacities_if_full_supported"], [6, 12, 18, 24, 30, 36, 42, 48])
            self.assertIn({"requests": 1, "rows": 5}, graphs["target_exact_low_concurrency_graphs"])

    def test_layout_describes_either_drafter_head_format(self):
        _, base = self.resolve("tp4")
        cluster = copy.deepcopy(base)
        cluster["environment"]["VLLM_DS41_DRAFT_NVFP4_HEAD"] = "1"
        draft = spark3.model_layout.describe(ROOT, cluster, 2097152)["drafter"]
        self.assertEqual(draft["head_formats"], {"lm_head": "nvfp4", "markov_output": "bf16"})
        self.assertEqual(draft["lm_head_nvfp4"]["swizzled_scales_e4m3_storage_shape"], [32384, 320])
        self.assertEqual(draft["lm_head_nvfp4"]["scale_alignment_extra_bytes"], 20480)
        self.assertIn("markov_output_bf16", draft)
        self.assertEqual([sorted(e) for e in draft["quantized_activation_examples"]],
                         [["lm_head", "rows_passed_to_head"]] * 3)

    def test_layout_rejects_incompatible_execution_flags(self):
        _, base = self.resolve("tp4")
        for key, value in (("VLLM_DS41_DRAFT_NVFP4_HEAD", "2"),
                           ("VLLM_DS41_MARKOV_NVFP4", "yes"),
                           ("SPARK3_DSPARK_MAIN_PROJ_TP", "0"),
                           ("VLLM_USE_V2_MODEL_RUNNER", "0"),
                           ("VLLM_ENABLE_ROCE_ALLREDUCE", "0"),
                           ("VLLM_DS41_BATCH_INVARIANT", "1")):
            cluster = copy.deepcopy(base)
            cluster["environment"][key] = value
            with self.assertRaises(ValueError):
                spark3.model_layout.describe(ROOT, cluster, 2097152)
        cluster = copy.deepcopy(base)
        spark3.topology.set_argument(cluster, "--engram-config", '{"projection_tp":false}')
        with self.assertRaisesRegex(ValueError, "Engram"):
            spark3.model_layout.describe(ROOT, cluster, 2097152)

    def test_layout_rejects_unaudited_sources_and_checkpoint(self):
        _, base = self.resolve("tp4")
        for field in ("source", "model"):
            cluster = copy.deepcopy(base)
            if field == "source":
                cluster["container"]["expected_labels"]["local.spark3.b12x.tree"] = "unknown"
            else:
                cluster["container"]["mounts"][0][0] = "/some/other/checkpoint"
            with self.assertRaises(ValueError):
                spark3.model_layout.describe(ROOT, cluster, 2097152)

    def test_wrong_topology_rejected(self):
        profile, base = self.resolve("tp3")
        with self.assertRaisesRegex(ValueError, "does not match"):
            profiles.materialize(profile, base, spark3.read_json("config/examples/nodes-ring4.json"), "nodes.json")

    def test_catalog_validation(self):
        original = spark3.read_json(profiles.DEFAULT_PROFILES)
        mutations = [
            ("tp4", lambda p: p.update(base_sha256="0" * 64), "base config changed"),
            ("tp4", lambda p: p["rocenante"].update(B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES="4194304"), "dispatch"),
            ("tp4", lambda p: p["rocenante"].update(B12X_ROCE_STREAM_CHUNK_BYTES="65536"), "exactly"),
            ("tp4", lambda p: p["nccl"].update(NCCL_BUFFSIZE="-1"), "positive integer"),
            ("tp4", lambda p: p["nccl"].update(NCCL_BUFFSIZE=None), "explicit values"),
            ("tp4", lambda p: p["nccl"].update(NCCL_THREAD_THRESHOLDS="1 2"), "six"),
            ("tp4", lambda p: p["nccl"].update(NCCL_SWITCHLESS_BIDIRECTIONAL="0"), "balanced TP4"),
            ("tp3", lambda p: p["rocenante"].update(B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES="1048576"), "does not provide"),
            ("tp4", lambda p: p.update(base_config="../elsewhere.json"), "relative"),
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "profiles.json"
            for name, mutate, message in mutations:
                with self.subTest(message=message):
                    catalog = copy.deepcopy(original)
                    mutate(catalog["profiles"][name])
                    path.write_text(json.dumps(catalog))
                    with self.assertRaisesRegex(ValueError, message):
                        profiles.resolve(ROOT, str(path.relative_to(ROOT)), name)

    def test_edited_channels_are_validated_against_ring_policy(self):
        profile, resolved = self.resolve("tp4")
        resolved["environment"]["NCCL_MIN_NCHANNELS"] = "2"
        with self.assertRaisesRegex(ValueError, "matching"):
            profiles.materialize(profile, resolved, spark3.read_json("config/examples/nodes-ring4.json"), "nodes.json")

    def test_cli_materializes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as directory:
            output = str((Path(directory) / "candidate.json").relative_to(ROOT))
            args = spark3.parser().parse_args([
                "tuning", "create", "tp4", "--nodes-config", "config/examples/nodes-ring4.json", "--output", output])
            with mock.patch("builtins.print"):
                self.assertEqual(args.func(args), 0)
            generated = spark3.read_json(output)
            self.assertEqual(generated["tuning_origin"]["profile"], "tp4")
            self.assertEqual(generated["tuning_origin"]["derived_layout"]["tensor_parallel_size"], 4)
            self.assertFalse(generated["deployment"]["launch_enabled"])
            with self.assertRaises(SystemExit):
                args.func(args)

    def test_cli_rejects_ambiguous_base_and_protected_output(self):
        for argv in (["--cluster-config", "config/cluster-4k.json", "tuning", "show", "tp4"],
                     ["tuning", "create", "tp4", "--nodes-config", "config/examples/nodes-ring4.json",
                      "--output", "config/new.json"]):
            args = spark3.parser().parse_args(argv)
            with self.assertRaises(SystemExit):
                args.func(args)


if __name__ == "__main__":
    unittest.main()
