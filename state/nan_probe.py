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


def _wrap_b12x_attention(owner: type[Any]) -> None:
    """Compare layer-0 B12X output with its reference on the live cache."""
    original = getattr(owner, "_run_attention")

    @functools.wraps(original)
    def checked(self: Any, *args: Any, **kwargs: Any) -> Any:
        prefix = getattr(self, "prefix", type(self).__name__)
        armed = os.path.exists(_ARM_FILE) and getattr(self, "layer_id", None) == 0
        expected = None
        if armed:
            from b12x.attention._shared.mla.compressed_reference import (
                compressed_sparse_mla_reference,
            )
            from vllm.models.deepseek_v41.nvidia.b12x_attention import (
                _flatten_cache,
            )

            q = kwargs["q"]
            indices = kwargs["swa_indices"]
            lengths = kwargs["swa_lengths"]
            sink = self.attn_sink
            cache = _flatten_cache(
                self.swa_cache_layer.kv_cache,
                name="DeepSeek V4.1 SWA cache",
            )
            expected = compressed_sparse_mla_reference(
                q,
                cache,
                indices,
                lengths,
                swa_page_size=self.swa_cache_layer.block_size,
                cache_format="deepseek_v41",
                sm_scale=self.scale,
                attn_sink=sink,
            )
            _check(f"{prefix}.b12x_reference.output", expected)
            valid = indices >= 0
            valid_indices = indices[valid]
            print(
                "spark3 NaN probe: live layer-0 B12X inputs "
                f"q={tuple(q.shape)} rows={int(q.shape[0])} "
                f"swa_width={int(indices.shape[1])} "
                f"lengths={lengths.detach().cpu().tolist()} "
                f"valid_indices={int(valid.sum().item())} "
                f"index_min={int(valid_indices.min().item()) if valid_indices.numel() else -1} "
                f"index_max={int(valid_indices.max().item()) if valid_indices.numel() else -1} "
                f"sink_finite={int(torch.isfinite(sink).sum().item())}/{sink.numel()} "
                f"reference_finite={int(torch.isfinite(expected).sum().item())}/{expected.numel()}",
                flush=True,
            )

        _check(f"{prefix}.b12x_run_attention.input", (args, kwargs))
        result = original(self, *args, **kwargs)
        output = kwargs["output"]
        if armed:
            torch.cuda.synchronize(output.device)
            finite = torch.isfinite(output)
            if not bool(finite.all().item()):
                assert expected is not None
                raise RuntimeError(
                    "spark3 NaN probe: B12X native output is non-finite while "
                    f"the live-cache reference is finite; shape={tuple(output.shape)}, "
                    f"native_nan={int(torch.isnan(output).sum().item())}, "
                    f"native_inf={int(torch.isinf(output).sum().item())}, "
                    f"reference_nan={int(torch.isnan(expected).sum().item())}, "
                    f"reference_inf={int(torch.isinf(expected).sum().item())}"
                )
        _check(f"{prefix}.b12x_run_attention.mutated", (args, kwargs))
        _check(f"{prefix}.b12x_run_attention.output", result)
        return result

    setattr(owner, "_run_attention", checked)


def _install() -> None:
    import vllm.models.deepseek_v41.nvidia.model as model
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention
    from vllm.models.deepseek_v41.nvidia.b12x_attention import (
        DeepseekV41B12xAttention,
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
    _wrap_b12x_attention(DeepseekV41B12xAttention)
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
