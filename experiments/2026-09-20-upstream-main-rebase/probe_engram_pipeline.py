#!/usr/bin/env python3
"""Replay the real layer-1 disk Engram rows through TP3 reduction and WKV.

This bounded diagnostic loads only the requested checkpoint rows and the
layer-1 WKV/q/k tensors. It compares RoCEnante with NCCL on the real rows,
then checks the actual WKV output and fused gate with finite test hidden states.
It does not construct the model or start a serving process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import triton
from b12x.loader._checkpoint import DirectWeightSession
from b12x.loader._pool import weight_allocation, weight_pool
from b12x.preparation import PreparationSession, PreparedCall
from transformers import PreTrainedTokenizerFast
from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
from vllm.model_executor.layers.quantization.modelopt import KMxfp8Static
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.weight_transfer import allocate_weights, weight_transfer
from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    good = values[finite]
    return {
        "shape": list(tensor.shape),
        "finite": int(finite.sum().item()),
        "total": tensor.numel(),
        "nan": int(torch.isnan(values).sum().item()),
        "inf": int(torch.isinf(values).sum().item()),
        "finite_min": float(good.min().item()) if good.numel() else None,
        "finite_max": float(good.max().item()) if good.numel() else None,
        "sha256": _sha256(tensor),
    }


def _quantize_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    blocked = source.float().reshape(rows, width // 32, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale_u8 = (exponent + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).float()
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _dequantize_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = scales.view(torch.float8_e8m0fnu).float()
    return (values.float().reshape(values.shape[0], -1, 32) * scale[..., None]).reshape(
        values.shape
    )


def _benchmark(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("engram_pipeline_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _local_checkpoint_rows(
    checkpoint_path: pathlib.Path,
    benchmark_path: pathlib.Path,
    prompt: str,
    rank: int,
) -> tuple[torch.Tensor, list[int], float, dict[str, Any]]:
    benchmark = _benchmark(benchmark_path)
    checkpoint = benchmark.Checkpoint(checkpoint_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint_path / "tokenizer.json")
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    token_map, compressed = benchmark.build_compressed_token_map(tokenizer)
    config = checkpoint.config["text_config"]
    if compressed != config["engram_compressed_vocab_size"]:
        raise ValueError("Engram tokenizer geometry differs from checkpoint")
    if 1 not in config["engram_layer_ids"]:
        raise ValueError("checkpoint has no layer-1 Engram")
    query = {
        "index": 0,
        "tokens": token_ids,
        "starts": [0, len(token_ids)],
        "history": [[-1, -1, -1]],
        "accepted_per_request": [len(token_ids)],
        "scheduled": len(token_ids),
        "accepted": len(token_ids),
    }
    case_args = SimpleNamespace(
        queue_depth=None,
        max_seqs=1,
        tp_size=3,
        tp_rank=rank,
        device=0,
        engram_resident_scales=False,
        engram_token_bound=True,
    )
    case = benchmark.make_case(
        case_args,
        checkpoint,
        "engram",
        1,
        len(token_ids),
        token_map,
        query,
        len(token_ids),
    )
    try:
        case.session.freeze()
        case.prepare(query)
        case.run(len(token_ids))
        torch.cuda.synchronize()
        correctness = case.check(query, len(token_ids), 256)
        rows = (
            case.out[: len(token_ids)].detach().clone().reshape(len(token_ids), 24, 256)
        )
        hash_ids = case.ids[: len(token_ids)].detach().clone()
        if not bool(torch.isfinite(rows).all().item()):
            raise RuntimeError("local disk Engram rows are non-finite")
        report = {
            "hash_sha256": _sha256(hash_ids),
            "local_rows": _summary(rows),
            "direct_checkpoint_rows_checked": len(correctness["checked_slices"]),
        }
    finally:
        case.close()
    checkpoint.unchanged()
    return rows, token_ids, float(config["rms_norm_eps"]), report


def _real_wkv_and_gate(
    checkpoint: pathlib.Path,
    rows: torch.Tensor,
    eps: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix = "layers.1.engram"
    index = checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    names = {
        role: f"{prefix}.{suffix}"
        for role, suffix in {
            "q": "q_weight",
            "k": "k_weight",
            "weight": "wkv.weight",
            "scale": "wkv.scale",
        }.items()
    }
    shards = {weight_map[name] for name in names.values()}
    if len(shards) != 1:
        raise ValueError(f"WKV tensors span unexpected shards: {shards}")
    allocation = weight_allocation(0)
    with (
        weight_pool(allocation=allocation, device=0) as allocator,
        DirectWeightSession(0, allocation_scope=allocator) as session,
        weight_transfer(session, allocator=allocator),
    ):
        sources = dict(
            session.weights(
                [checkpoint / shards.pop()], prefixes=(prefix,), index_path=index
            )
        )
        if set(names.values()) - sources.keys():
            raise ValueError("WKV checkpoint tensors are missing")

        def parameter(role: str, shape: tuple[int, ...], dtype: torch.dtype):
            return torch.nn.Parameter(
                allocate_weights(torch.empty, shape, dtype=dtype, device="cuda"),
                requires_grad=False,
            )

        q = parameter("q", sources[names["q"]].shape, sources[names["q"]].dtype)
        k = parameter("k", sources[names["k"]].shape, sources[names["k"]].dtype)
        weight = parameter(
            "weight", sources[names["weight"]].shape, sources[names["weight"]].dtype
        )
        raw_scale = sources[names["scale"]]
        scale = parameter(
            "scale",
            (int(raw_scale.shape[0]) * 32, int(raw_scale.shape[1])),
            torch.uint8,
        )
        default_weight_loader(q, sources[names["q"]])
        default_weight_loader(k, sources[names["k"]])
        default_weight_loader(weight, sources[names["weight"]])
        KMxfp8Static.get_scale_weight_loader(
            default_weight_loader, SimpleNamespace(scale_block_size=(32, 32))
        )(scale, raw_scale)
        session.flush()
        torch.cuda.synchronize()
        reference_rows = 32
        reference_weight = weight[:reference_rows].detach().clone()
        reference_scale = scale[:reference_rows].detach().clone()
        loaded_report = {
            "q_sha256": _sha256(q),
            "k_sha256": _sha256(k),
            "weight_sha256": _sha256(weight),
            "scale_sha256": _sha256(scale),
            "scale_ff_bytes": int((scale == 0xFF).sum().item()),
            "io": session.stats(),
        }

        layer = torch.nn.Module()
        layer.prefix = "diagnostic.layers.1.engram.wkv"
        layer.weight = weight
        layer.weight_scale = scale
        kernel = object.__new__(B12xMxfp8LinearKernel)
        kernel.process_weights_after_loading(layer)
        packed = layer.b12x_mxfp8_packed_weight.weight
        loaded_report["packed_weight_sha256"] = _sha256(packed.values)
        loaded_report["packed_scale_sha256"] = _sha256(packed.scale_mma)
        unit = kernel.get_b12x_pre_profile_unit(
            layer, tuple(sorted({1, rows.shape[0]})), torch.bfloat16
        )
        unit.compile()
        kv = kernel.apply_weights(layer, rows.flatten(1).contiguous())
        torch.cuda.synchronize()
        kv_report = _summary(kv)
        if kv_report["finite"] != kv_report["total"]:
            raise RuntimeError("real-row WKV output is non-finite")
        input_values, input_scales = _quantize_rows(rows.flatten(1).contiguous())
        expected = (
            _dequantize_rows(input_values, input_scales)
            @ _dequantize_rows(reference_weight, reference_scale).T
        ).to(kv.dtype)
        selected = kv[:, :reference_rows]
        kv_report["reference"] = {
            "rows": reference_rows,
            "expected_sha256": _sha256(expected),
            "actual_sha256": _sha256(selected),
            "exact_bf16": bool(torch.equal(selected, expected)),
            "max_abs_error": float(
                (selected.float() - expected.float()).abs().max().item()
            ),
        }

        torch.manual_seed(20260923)
        hidden = (
            torch.randn(rows.shape[0], 4, 5120, dtype=torch.bfloat16, device="cuda") / 8
        ).contiguous()
        mask = torch.ones(rows.shape[0], dtype=torch.bool, device="cuda")
        output = torch.empty_like(hidden)
        _fused_engram_post_wkv_kernel[(rows.shape[0] * 4,)](
            hidden,
            kv,
            q,
            k,
            mask,
            output,
            rows.shape[0],
            *hidden.stride(),
            *kv.stride(),
            *q.stride(),
            *k.stride(),
            mask.stride(0),
            *output.stride(),
            eps,
            1e-6,
            DIM=5120,
            HC_MULT=4,
            BLOCK_SIZE=triton.next_power_of_2(5120),
            HAS_MASK=True,
            num_warps=8,
        )
        torch.cuda.synchronize()
        gate_report = _summary(output)
        if gate_report["finite"] != gate_report["total"]:
            raise RuntimeError("real-row Engram gate output is non-finite")
        return loaded_report, kv_report, gate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=pathlib.Path("/models")
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        default=pathlib.Path(
            "/opt/spark3/candidate/b12x/benchmarks/benchmark_ngram_ssd.py"
        ),
    )
    parser.add_argument("--prompt", default="1 + 1 =")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 3:
        raise ValueError(f"this diagnostic requires TP3, got {world}")
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    device_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=90))
    runtime = None
    try:
        from b12x.comm import roce
        from b12x.comm.roce import _preparation

        runtime = roce.AllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda", 0),
            max_size=2 << 20,
            max_gather_bytes=4 << 20,
        )
        query = roce.query_from_runtime(
            runtime,
            surface="AllReduce.all_reduce",
            call={"dtypes": ("float16", "bfloat16", "float32")},
            topology="roce_rdma",
            peer_hosts=tuple(f"rank:{peer}" for peer in range(world)),
        )
        plan = roce.plan(query, runtime=runtime)
        seeds = [
            torch.zeros(16 // dtype.itemsize, dtype=dtype, device="cuda")
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ]

        def prepare(state: object) -> PreparedCall:
            calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
            calls.append(_preparation.prepared_gather_call(state, inp=seeds[1]))

            def prime() -> list[object]:
                dist.barrier()
                return [call.run() for call in calls]

            return PreparedCall(
                run=prime,
                output=tuple(call.output for call in calls),
                owners=(*seeds, *calls),
            )

        request = plan.request(
            name="diagnostic.engram.real-pipeline", prepare_call=prepare
        )
        with PreparationSession(
            device=torch.device("cuda", 0), autotune=False, compile_workers=1
        ) as preparation:
            preparation.prepare((request,))
            local, token_ids, eps, local_report = _local_checkpoint_rows(
                args.checkpoint, args.benchmark, args.prompt, rank
            )
            dist.barrier()
            expected = local.clone()
            dist.all_reduce(expected, group=device_group)
            actual = runtime.all_reduce(local, plan=plan)
            torch.cuda.synchronize()
            runtime.check_health()
            equal = bool(torch.equal(actual, expected))
            reduced_report = _summary(actual)
            if not equal or reduced_report["finite"] != reduced_report["total"]:
                raise RuntimeError("real-row RoCEnante reduction differs from NCCL")
            loaded_report, kv_report, gate_report = _real_wkv_and_gate(
                args.checkpoint, actual, eps
            )
            report = {
                "rank": rank,
                "token_ids": token_ids,
                "local": local_report,
                "reduced_equal_nccl": equal,
                "reduced": reduced_report,
                "loaded": loaded_report,
                "wkv": kv_report,
                "gate": gate_report,
                "runtime_stats": runtime.stats(),
            }
            print(json.dumps(report, sort_keys=True), flush=True)
            dist.barrier()
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group(device_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
