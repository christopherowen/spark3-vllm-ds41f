#!/usr/bin/env python3
"""Validate one native Engram WKV through the production B12X load path.

This deliberately avoids constructing the full model.  It loads the selected
Engram q/k and ModelOpt MXFP8 WKV tensors into B12X-owned allocations, applies
the checkpoint's 32-row scale expansion, packs the weight with the serving
kernel, and compares a small projection against an independent dequantized
reference.  The largest live checkpoint tensor is the roughly 150 MiB WKV;
no expert weights, KV cache, or distributed process group are created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from types import SimpleNamespace

import torch
from b12x.loader._checkpoint import DirectWeightSession
from b12x.loader._pool import weight_allocation, weight_pool
from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
from vllm.model_executor.layers.quantization.modelopt import KMxfp8Static
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.weight_transfer import allocate_weights, weight_transfer


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--reference-rows", type=int, default=32)
    args = parser.parse_args()

    if args.tokens < 1 or args.reference_rows < 1:
        parser.error("tokens and reference-rows must be positive")
    torch.cuda.set_device(args.device)

    prefix = f"layers.{args.layer}.engram"
    index_path = args.checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
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
        raise ValueError(f"Engram WKV inputs span unexpected shards: {shards}")
    shard = args.checkpoint / shards.pop()

    allocation = weight_allocation(args.device)
    with (
        weight_pool(allocation=allocation, device=args.device) as allocator,
        DirectWeightSession(
            args.device,
            allocation_scope=allocator,
        ) as session,
        weight_transfer(session, allocator=allocator),
    ):
        sources = dict(
            session.weights(
                [shard],
                prefixes=(prefix,),
                index_path=index_path,
            )
        )
        missing = set(names.values()) - sources.keys()
        if missing:
            raise ValueError(f"missing checkpoint tensors: {sorted(missing)}")

        q = torch.nn.Parameter(
            allocate_weights(
                torch.empty,
                sources[names["q"]].shape,
                dtype=sources[names["q"]].dtype,
                device="cuda",
            ),
            requires_grad=False,
        )
        k = torch.nn.Parameter(
            allocate_weights(
                torch.empty,
                sources[names["k"]].shape,
                dtype=sources[names["k"]].dtype,
                device="cuda",
            ),
            requires_grad=False,
        )
        weight = torch.nn.Parameter(
            allocate_weights(
                torch.empty,
                sources[names["weight"]].shape,
                dtype=sources[names["weight"]].dtype,
                device="cuda",
            ),
            requires_grad=False,
        )
        raw_scale = sources[names["scale"]]
        scale_shape = (int(raw_scale.shape[0]) * 32, int(raw_scale.shape[1]))
        scale = torch.nn.Parameter(
            allocate_weights(
                torch.empty,
                scale_shape,
                dtype=torch.uint8,
                device="cuda",
            ),
            requires_grad=False,
        )

        default_weight_loader(q, sources[names["q"]])
        default_weight_loader(k, sources[names["k"]])
        default_weight_loader(weight, sources[names["weight"]])
        scale_loader = KMxfp8Static.get_scale_weight_loader(
            default_weight_loader,
            SimpleNamespace(scale_block_size=(32, 32)),
        )
        scale_loader(scale, raw_scale)
        session.flush()
        torch.cuda.synchronize()

        reference_rows = min(args.reference_rows, int(weight.shape[0]))
        reference_weight = weight[:reference_rows].detach().clone()
        reference_scale = scale[:reference_rows].detach().clone()
        weight_sample = weight.reshape(-1)[:: max(1, weight.numel() // 65536)]
        q_cpu = q.detach().cpu()
        k_cpu = k.detach().cpu()
        scale_u8 = scale.detach().view(torch.uint8)
        loaded = {
            "q_sha256": _sha256(q.detach()),
            "k_sha256": _sha256(k.detach()),
            "q_finite": int(torch.isfinite(q_cpu).sum().item()),
            "q_values": q.numel(),
            "k_finite": int(torch.isfinite(k_cpu).sum().item()),
            "k_values": k.numel(),
            "weight_sample_finite": int(
                torch.isfinite(weight_sample.float()).sum().item()
            ),
            "weight_sample_values": weight_sample.numel(),
            "scale_min": int(scale_u8.min().item()),
            "scale_max": int(scale_u8.max().item()),
            "scale_ff_bytes": int((scale_u8 == 0xFF).sum().item()),
            "io": session.stats(),
        }

        layer = torch.nn.Module()
        layer.prefix = f"diagnostic.{prefix}.wkv"
        layer.weight = weight
        layer.weight_scale = scale
        kernel = object.__new__(B12xMxfp8LinearKernel)
        kernel.process_weights_after_loading(layer)
        unit = kernel.get_b12x_pre_profile_unit(
            layer,
            tuple(sorted({1, args.tokens})),
            torch.bfloat16,
        )
        unit.compile()

        torch.manual_seed(20260923)
        source = (
            torch.randn(
                (args.tokens, int(reference_weight.shape[1])),
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 8
        ).contiguous()
        source_values, source_scales = _quantize_rows(source)
        expected = (
            _dequantize_rows(source_values, source_scales)
            @ _dequantize_rows(
                reference_weight,
                reference_scale,
            ).T
        )
        actual = kernel.apply_weights(layer, source)
        torch.cuda.synchronize()
        selected = actual[:, :reference_rows]
        finite = torch.isfinite(actual)
        max_abs_error = float(
            (selected.float() - expected.to(selected.dtype).float()).abs().max().item()
        )
        exact = bool(torch.equal(selected, expected.to(selected.dtype)))

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "layer": args.layer,
                "allocation": allocation,
                "loaded": loaded,
                "projection": {
                    "shape": list(actual.shape),
                    "finite_values": int(finite.sum().item()),
                    "total_values": actual.numel(),
                    "nan_values": int(torch.isnan(actual).sum().item()),
                    "inf_values": int(torch.isinf(actual).sum().item()),
                    "reference_rows": reference_rows,
                    "reference_exact_bf16": exact,
                    "max_abs_error": max_abs_error,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
