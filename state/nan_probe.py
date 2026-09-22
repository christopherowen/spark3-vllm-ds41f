#!/usr/bin/env python3
"""Fail at the first non-finite DeepSeek V4.1 activation when armed.

This is an opt-in diagnostic entry point.  Startup and warmup run normally;
creating ``/tmp/spark3-nan-probe`` inside a live container arms checks for the
next request.  The wrapper then launches the ordinary vLLM CLI in-process.
"""

from __future__ import annotations

import functools
import os
import runpy
import sys
from collections.abc import Iterator
from typing import Any

import torch


_ARM_FILE = "/tmp/spark3-nan-probe"


class _NoPdlPlatform:
    """Delegate every platform query except PDL capability."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def is_arch_support_pdl(self) -> bool:
        return False


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif (
        type(value).__name__ == "QuantizedActivation"
        and hasattr(value, "data")
        and hasattr(value, "scale")
    ):
        yield from _tensors(value.data)
        yield from _tensors(value.scale)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _check(stage: str, value: Any) -> None:
    if not os.path.exists(_ARM_FILE):
        return
    for index, tensor in enumerate(_tensors(value)):
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        raise RuntimeError(
            "spark3 NaN probe: first non-finite activation at "
            f"{stage}[{index}], shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, nan={nan_count}, inf={inf_count}"
        )


def _wrap_method(
    owner: type[Any],
    method_name: str,
    label: str,
    *,
    check_mutated_inputs: bool = False,
) -> None:
    original = getattr(owner, method_name)

    @functools.wraps(original)
    def checked(self: Any, *args: Any, **kwargs: Any) -> Any:
        prefix = getattr(self, "prefix", type(self).__name__)
        _check(f"{prefix}.{label}.input", (args, kwargs))
        output = original(self, *args, **kwargs)
        if check_mutated_inputs:
            _check(f"{prefix}.{label}.mutated", (args, kwargs))
        _check(f"{prefix}.{label}.output", output)
        return output

    setattr(owner, method_name, checked)


def _wrap_function(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)

    @functools.wraps(original)
    def checked(*args: Any, **kwargs: Any) -> Any:
        _check(f"{function_name}.input", (args, kwargs))
        output = original(*args, **kwargs)
        _check(f"{function_name}.output", output)
        return output

    setattr(module, function_name, checked)


def _wrap_sparse_entry(owner: type[Any]) -> None:
    """Reconstruct layer-0 Q sequentially at the eager attention boundary."""
    original = getattr(owner, "_sparse_indexer_and_attn")

    @functools.wraps(original)
    def checked(self: Any, *args: Any, **kwargs: Any) -> Any:
        prefix = getattr(self, "prefix", type(self).__name__)
        if os.path.exists(_ARM_FILE) and getattr(self, "layer_id", None) == 0:
            hidden_states = args[0]
            positions = args[6]
            qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
            _check(f"{prefix}.diagnostic_fused_wqa_wkv.output", qr_kv)
            qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
            _check(f"{prefix}.diagnostic_split_qkv_norm.output", (qr, qr_scale, kv))
            q = self._wq_b_proj(qr, qr_scale).view(
                -1, self.n_local_heads, self.head_dim
            )
            _check(f"{prefix}.diagnostic_wq_b.output", q)
            from vllm.forward_context import get_forward_context

            q = self._fused_qnorm_rope_kv_insert(
                q,
                kv,
                positions,
                get_forward_context().attn_metadata,
            )
            _check(f"{prefix}.diagnostic_q_rope_kv_insert.output", q)
        _check(f"{prefix}.sparse_indexer_and_attn.input", (args, kwargs))
        output = original(self, *args, **kwargs)
        _check(f"{prefix}.sparse_indexer_and_attn.mutated", (args, kwargs))
        _check(f"{prefix}.sparse_indexer_and_attn.output", output)
        return output

    setattr(owner, "_sparse_indexer_and_attn", checked)


def _install() -> None:
    import vllm.models.common.ops.fused_qk_rmsnorm as fused_rmsnorm
    import vllm.models.deepseek_v41.nvidia.model as model
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention
    from vllm.models.deepseek_v41.nvidia.b12x_attention import (
        DeepseekV41B12xAttention,
    )

    # The preceding MXFP8 producer selected on SM121 does not establish the
    # programmatic grid dependency expected by this consumer.  Keep every
    # other platform capability intact while testing the suspected RAW race.
    fused_rmsnorm.current_platform = _NoPdlPlatform(
        fused_rmsnorm.current_platform
    )

    _wrap_method(VocabParallelEmbedding, "forward", "embedding")
    for method_name in (
        "_run_parallel_input_projections",
        "_split_qkv_and_norm",
        "_wq_b_proj",
        "_fused_qnorm_rope_kv_insert",
    ):
        _wrap_method(
            DeepseekV4Attention,
            method_name,
            method_name.removeprefix("_"),
        )
    _wrap_sparse_entry(DeepseekV4Attention)
    _wrap_method(
        DeepseekV41B12xAttention,
        "_run_attention",
        "b12x_run_attention",
        check_mutated_inputs=True,
    )
    _wrap_method(
        DeepseekV41B12xAttention,
        "forward_mqa",
        "b12x_forward_mqa",
        check_mutated_inputs=True,
    )
    _wrap_method(DeepseekV41B12xAttention, "_o_proj", "o_proj")
    _wrap_method(DeepseekV4Attention, "forward", "attention")
    _wrap_method(model.DeepseekV4MoE, "forward", "moe")
    _wrap_method(model.DeepseekV4DecoderLayer, "forward", "decoder")
    _wrap_method(LogitsProcessor, "_get_logits", "logits")
    for name in (
        "mhc_pre_delayed_tilelang",
        "mhc_shifted_post_pre",
        "mhc_post_tilelang",
        "hc_collapse_triton",
    ):
        _wrap_function(model, name)


_install()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
