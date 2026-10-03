#!/usr/bin/env python3
"""Check derived shapes against pure helpers from the exact audited Git trees.

No Torch/CUDA imports, node access or GPU work. Git repositories must already
contain the audited source trees; this command never prepares or changes them.
"""

import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import model_layout
import transport_profiles


def source(repo, tree, path):
    return subprocess.check_output(["git", "-C", repo, "show", f"{tree}:{path}"], text=True)


def function(code, name, namespace):
    node = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), name, "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-repo")
    parser.add_argument("--b12x-repo")
    args = parser.parse_args()
    if not args.vllm_repo or not args.b12x_repo:
        loader = importlib.machinery.SourceFileLoader("layout_build", str(ROOT / "bin/spark3"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        build = importlib.util.module_from_spec(spec)
        loader.exec_module(build)
        _, cluster = transport_profiles.resolve(ROOT, transport_profiles.DEFAULT_PROFILES, "tp4")
        directory = build.build_directory(build.build_inputs(build.read_json(cluster["upstreams_config"]))) / "src"
        args.vllm_repo = args.vllm_repo or str(directory / "vllm")
        args.b12x_repo = args.b12x_repo or str(directory / "b12x")
    results = {}
    for name in ("tp3", "tp4"):
        _, cluster = transport_profiles.resolve(ROOT, transport_profiles.DEFAULT_PROFILES, name)
        report = model_layout.describe(ROOT, cluster, 2097152)
        trees = report["source_trees"]
        paths = {
            "vllm": ["vllm/_custom_ops.py", "vllm/models/deepseek_v4_1/sp_prefill.py",
                     "vllm/v1/worker/gpu/cudagraph_utils.py", "vllm/models/deepseek_v4_1/nvidia/dspark.py",
                     "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
                     "vllm/models/deepseek_v4_1/attention.py", "vllm/model_executor/models/config.py",
                     "vllm/model_executor/layers/logits_processor.py",
                     "vllm/models/deepseek_v4_1/common/engram.py", "vllm/model_executor/models/qwen3_dspark.py",
                     "vllm/model_executor/layers/vocab_parallel_embedding.py",
                     "vllm/model_executor/layers/quantization/online/nvfp4.py",
                     "vllm/model_executor/kernels/linear/nvfp4/b12x.py"],
            "b12x": ["b12x/_lib/intrinsics.py", "b12x/gemm/blockscaled/_a16.py",
                     "b12x/comm/roce/roce_oneshot.py", "b12x/moe/_shared/kernels/w4a8_compact_micro.py"],
        }
        sources = {project: {path: source(getattr(args, f"{project}_repo"), trees[project], path)
                             for path in project_paths} for project, project_paths in paths.items()}
        custom_ops = sources["vllm"]["vllm/_custom_ops.py"]
        fake_torch = SimpleNamespace(empty=lambda shape, **kwargs: tuple(shape), int32="int32", uint8="uint8")
        ns = {"torch": fake_torch}
        function(custom_ops, "create_fp4_scale_tensor", ns)
        quant = function(custom_ops, "create_fp4_output_tensors", ns)
        # Only heads the profile stores as NVFP4 have packed layouts to check.
        draft = report["drafter"]
        layouts = [draft[key] for key in ("lm_head_nvfp4", "markov_output_nvfp4") if key in draft]
        for example in draft["quantized_activation_examples"]:
            layouts.extend(example[key] for key in ("lm_head", "markov_output") if key in example)
        with mock.patch.dict(sys.modules, {"vllm.utils.math_utils": SimpleNamespace(round_up=model_layout.round_up)}):
            for layout in layouts:
                values, scales = quant(*layout["logical_shape"], "cpu", True)
                assert list(values) == layout["packed_values_uint8_shape"]
                assert [scales[0], scales[1] * 4] == layout["swizzled_scales_e4m3_storage_shape"]
        sp = function(sources["vllm"]["vllm/models/deepseek_v4_1/sp_prefill.py"], "rows_above_bytes", {})
        tp = report["tensor_parallel_size"]
        assert sp(2097152, 5120 * 2, tp) == report["scheduled_rows"]["prefill_sp_min_live_rows"]
        graphs = function(sources["vllm"]["vllm/v1/worker/gpu/cudagraph_utils.py"], "dense_varlen_decode_shapes",
                          {"_DENSE_VARLEN_DECODE_MAX_REQS": 2})
        assert [{"requests": r, "rows": t} for r, t in graphs(8, 6, 48)] == report["scheduled_rows"]["target_exact_low_concurrency_graphs"]
        gather = function(sources["b12x"]["b12x/comm/roce/roce_oneshot.py"], "_direct_gather_layout", {"PACK_BYTES": 16})
        for rows, width, dim, pointer, expected in (
            (3, 5, 1, 0, False), (8, 5, 0, 0, True), (8, 5, 1, 0, False),
            (3, 8, 1, 0, True), (3, 8, 1, 2, False),
        ):
            tensor = SimpleNamespace(shape=(rows, width), numel=lambda: rows * width,
                                     element_size=lambda: 2, data_ptr=lambda: pointer)
            assert gather(None, tensor, dim) == expected
        results[name] = {
            "source_trees": trees, "nvfp4_storage_cases_passed": len(layouts),
            "sp_boundary_passed": True, "exact_target_graph_shapes_passed": True,
            "direct_gather_alignment_cases_passed": 5,
            "file_sha256": {project: {path: hashlib.sha256(code.encode()).hexdigest() for path, code in files.items()}
                            for project, files in sources.items()},
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
