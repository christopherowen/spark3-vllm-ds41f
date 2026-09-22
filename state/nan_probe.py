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


def _install() -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mxfp8
    import vllm.models.deepseek_v41.nvidia.model as model
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention
    from vllm.models.deepseek_v41.nvidia.b12x_attention import (
        DeepseekV41B12xAttention,
    )

    # B12X accepts an optional launch stream, but its prepared MXFP8 path
    # currently forwards ``None`` to the CuTe GEMM launcher.  Its Triton
    # quantizer and the following vLLM kernels use PyTorch's current stream.
    # Pass that stream explicitly to test whether the corrupt Q tensor is a
    # producer/consumer race at this boundary rather than bad arithmetic.
    def apply_mxfp8_on_current_stream(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        packed_weight = layer.b12x_mxfp8_packed_weight
        plan = getattr(layer, "b12x_mxfp8_plan", None)
        if plan is None or plan.prepared is None:
            raise RuntimeError(
                "b12x MXFP8 linear plan must be prepared before memory "
                "profiling or capture"
            )
        input_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output_shape = [*x.shape[:-1], int(packed_weight.out_features)]
        mxfp8 = b12x_mxfp8._import_b12x_mxfp8()
        assert mxfp8 is not None
        output = mxfp8.mm(
            input_2d,
            packed_weight,
            plan=plan,
            bias=bias,
            stream=torch.cuda.current_stream(input_2d.device),
        )
        return output.view(*output_shape)

    b12x_mxfp8._apply_b12x_mxfp8_packed_linear = apply_mxfp8_on_current_stream

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
