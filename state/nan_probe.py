#!/usr/bin/env python3
"""Fail at the first non-finite DeepSeek V4.1 activation when armed.

This is an opt-in diagnostic entry point.  Startup and warmup run normally;
creating ``/tmp/spark3-nan-probe`` inside a live container arms checks for the
next request.  The wrapper then launches the ordinary vLLM CLI in-process.
"""

from __future__ import annotations

import functools
import hashlib
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


def _trace(stage: str, tensor: torch.Tensor) -> None:
    if not os.path.exists(_ARM_FILE):
        return
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    good = values[finite]
    print(
        "spark3 NaN probe: "
        f"{stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={int(finite.sum().item())}/{tensor.numel()} "
        f"min={float(good.min().item()) if good.numel() else None} "
        f"max={float(good.max().item()) if good.numel() else None}",
        flush=True,
    )


def _wrap_engram_prepare(owner: type[Any]) -> None:
    original = owner.prepare_disk

    @functools.wraps(original)
    def checked(self: Any, hash_ids: torch.Tensor) -> None:
        armed = os.path.exists(_ARM_FILE)
        stage = f"Engram[{self.layer_hash_index}]"
        print(
            f"spark3 NaN probe: {stage}.prepare_disk invoked "
            f"armed={armed} tokens={hash_ids.shape[0]}",
            flush=True,
        )
        if armed:
            raw = hash_ids.detach().contiguous().view(torch.uint8).cpu()
            digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
            print(
                f"spark3 NaN probe: {stage}.hash_ids "
                f"shape={tuple(hash_ids.shape)} sha256={digest}",
                flush=True,
            )
        original(self, hash_ids)
        if armed:
            rows = self.staged_rows[: hash_ids.shape[0]]
            _trace(f"{stage}.prepared_rows", rows)
            bad = ~torch.isfinite(rows)
            bad_heads = bad.any(dim=-1).nonzero()
            details = [
                (
                    int(token),
                    int(head),
                    int(hash_ids[token, head].item()),
                    int(bad[token, head].sum().item()),
                )
                for token, head in bad_heads[:16].tolist()
            ]
            table = self.embed_tokens.disk_table
            cache_values = table.weight[: hash_ids.numel()]
            cache_scales = table.scale_bytes[: hash_ids.numel()]
            print(
                f"spark3 NaN probe: {stage}.bad_heads "
                f"count={bad_heads.shape[0]} first16={details} "
                f"cache_value_nan={int(torch.isnan(cache_values.float()).sum().item())} "
                f"cache_scale_ff={int((cache_scales == 255).sum().item())}",
                flush=True,
            )

    owner.prepare_disk = checked


def _wrap_model_state(owner: type[Any], default_owner: type[Any]) -> None:
    original_init = owner.__init__
    original_prepare = owner.prepare_inputs
    base_prepare = default_owner.prepare_inputs

    @functools.wraps(original_init)
    def checked_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        print(
            "spark3 NaN probe: model_state initialized "
            f"type={type(self).__name__} "
            f"lookback={None if self.lookback_token_ids is None else tuple(self.lookback_token_ids.shape)} "
            f"disk_engram_models={len(self.disk_engram_models)}",
            flush=True,
        )

    @functools.wraps(original_prepare)
    def checked_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.path.exists(_ARM_FILE):
            print(
                "spark3 NaN probe: custom model_state.prepare_inputs "
                f"lookback_present={self.lookback_token_ids is not None} "
                f"disk_engram_models={len(self.disk_engram_models)}",
                flush=True,
            )
        return original_prepare(self, *args, **kwargs)

    @functools.wraps(base_prepare)
    def checked_base_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.path.exists(_ARM_FILE):
            print(
                "spark3 NaN probe: base model_state.prepare_inputs "
                f"type={type(self).__name__}",
                flush=True,
            )
        return base_prepare(self, *args, **kwargs)

    owner.__init__ = checked_init
    owner.prepare_inputs = checked_prepare
    default_owner.prepare_inputs = checked_base_prepare


def _wrap_engram(owner: type[Any]) -> None:
    original = owner.forward

    @functools.wraps(original)
    def checked(
        self: Any,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if os.path.exists(_ARM_FILE):
            stage = getattr(self, "prefix", "Engram")
            staged = self.staged_rows[: hash_ids.shape[0]]
            _check(f"{stage}.staged_rows", staged)
            _check(f"{stage}.hidden", hidden_states)
            _trace(f"{stage}.staged_rows", staged)
            _trace(f"{stage}.hidden", hidden_states)
        output = original(self, hidden_states, hash_ids, token_mask)
        _check(f"{getattr(self, 'prefix', 'Engram')}.output", output)
        _trace(f"{getattr(self, 'prefix', 'Engram')}.output", output)
        return output

    owner.forward = checked


def _wrap_b12x_wkv(owner: type[Any]) -> None:
    original = owner.apply_weights

    @functools.wraps(original)
    def checked(self: Any, layer: Any, x: torch.Tensor, bias: Any = None) -> Any:
        stage = getattr(layer, "prefix", "")
        selected = os.path.exists(_ARM_FILE) and stage.endswith(".engram.wkv")
        if selected:
            _check(f"{stage}.input", x)
            _trace(f"{stage}.input", x)
        output = original(self, layer, x, bias)
        if selected:
            _check(f"{stage}.output", output)
            _trace(f"{stage}.output", output)
        return output

    owner.apply_weights = checked


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
        if label == "embed" and isinstance(output, torch.Tensor):
            _trace(f"{prefix}.{label}.output", output)
        if label in ("attention", "moe", "decoder", "logits", "o_proj"):
            for index, tensor in enumerate(_tensors(output)):
                if index >= (4 if label == "decoder" else 1):
                    break
                if tensor.is_floating_point():
                    _trace(f"{prefix}.{label}.output[{index}]", tensor)
        return output

    setattr(owner, method_name, checked)


def _wrap_function(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    calls = 0

    @functools.wraps(original)
    def checked(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        selected = (
            function_name == "mhc_shifted_post_pre"
            and os.path.exists(_ARM_FILE)
            and calls < 8
        )
        reference = None
        if selected:
            calls += 1
            x, residual, post_mix, res_mix = args[:4]
            reference = (
                torch.einsum("tij,tih->tjh", res_mix.float(), residual.float())
                + post_mix.float() * x.float().unsqueeze(-2)
            ).to(residual.dtype)
        _check(f"{function_name}.input", (args, kwargs))
        output = original(*args, **kwargs)
        _check(f"{function_name}.output", output)
        if selected:
            actual = output[0]
            error = (actual.float() - reference.float()).abs()
            print(
                "spark3 NaN probe: mhc_shifted_post_pre "
                f"call={calls} x_max={float(x.float().abs().max().item())} "
                f"residual_max={float(residual.float().abs().max().item())} "
                f"post_mix_max={float(post_mix.float().abs().max().item())} "
                f"res_mix_max={float(res_mix.float().abs().max().item())} "
                f"actual_max={float(actual.float().abs().max().item())} "
                f"reference_max={float(reference.float().abs().max().item())} "
                f"max_abs_error={float(error.max().item())}",
                flush=True,
            )
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
    """Compare early-layer B12X outputs with the live-cache reference."""
    original = getattr(owner, "_run_attention")

    @functools.wraps(original)
    def checked(self: Any, *args: Any, **kwargs: Any) -> Any:
        prefix = getattr(self, "prefix", type(self).__name__)
        if os.path.exists(_ARM_FILE) and kwargs.get("indexed_indices") is not None:
            from vllm.forward_context import get_forward_context

            assert self.compressed_cache_prefix is not None
            compressed = get_forward_context().attn_metadata[
                self.compressed_cache_prefix
            ]
            if getattr(self, "layer_id", None) == 2:
                print(
                    "spark3 NaN probe: "
                    f"{prefix}.indexed_block_table "
                    f"swa_first={kwargs['block_table'][0, :4].detach().cpu().tolist()} "
                    f"compressed_first={compressed.block_table[0, :4].detach().cpu().tolist()}",
                    flush=True,
                )
            kwargs = {**kwargs, "block_table": compressed.block_table}
        armed = os.path.exists(_ARM_FILE) and getattr(self, "layer_id", None) in (
            0,
            1,
            2,
        )
        expected = None
        if armed:
            from b12x.attention._shared.mla.compressed_reference import (
                compressed_sparse_mla_reference,
            )
            from vllm.models.deepseek_v41.nvidia.b12x_attention import (
                _flatten_cache,
            )
            from vllm.models.deepseek_v41.common.ops import (
                compute_global_topk_indices_and_lens,
            )

            q = kwargs["q"]
            indices = kwargs["swa_indices"]
            lengths = kwargs["swa_lengths"]
            sink = self.attn_sink
            cache = _flatten_cache(
                self.swa_cache_layer.kv_cache,
                name="DeepSeek V4.1 SWA cache",
            )
            extra: dict[str, Any] = {}
            indexed_indices = kwargs["indexed_indices"]
            if indexed_indices is not None:
                indexed_cache = kwargs["indexed_cache"]
                assert indexed_cache is not None
                indexed_page_size = (
                    self._vllm_config.cache_config.block_size // self.compress_ratio
                )
                physical_indices, indexed_lengths = (
                    compute_global_topk_indices_and_lens(
                        indexed_indices,
                        kwargs["token_to_req_indices"],
                        kwargs["block_table"],
                        indexed_page_size,
                        kwargs["is_valid_token"],
                    )
                )
                extra = {
                    "extra_k_cache": indexed_cache,
                    "extra_indices": physical_indices,
                    "extra_topk_lengths": indexed_lengths,
                    "extra_page_size": indexed_page_size,
                }
            expected = compressed_sparse_mla_reference(
                q,
                cache,
                indices,
                lengths,
                swa_page_size=self.swa_cache_layer.block_size,
                cache_format="deepseek_v41",
                sm_scale=self.scale,
                attn_sink=sink,
                **extra,
            )
            _check(f"{prefix}.b12x_reference.output", expected)
            valid = indices >= 0
            valid_indices = indices[valid]
            print(
                f"spark3 NaN probe: {prefix}.b12x_inputs "
                f"q={tuple(q.shape)} rows={int(q.shape[0])} "
                f"swa_width={int(indices.shape[1])} "
                f"lengths={lengths.detach().cpu().tolist()} "
                f"valid_indices={int(valid.sum().item())} "
                f"index_min={int(valid_indices.min().item()) if valid_indices.numel() else -1} "
                f"index_max={int(valid_indices.max().item()) if valid_indices.numel() else -1} "
                f"sink_finite={int(torch.isfinite(sink).sum().item())}/{sink.numel()} "
                f"indexed_lengths={indexed_lengths.detach().cpu().tolist() if indexed_indices is not None else None} "
                f"reference_finite={int(torch.isfinite(expected).sum().item())}/{expected.numel()}",
                flush=True,
            )

        _check(f"{prefix}.b12x_run_attention.input", (args, kwargs))
        result = original(self, *args, **kwargs)
        output = kwargs["output"]
        if armed:
            torch.cuda.synchronize(output.device)
            finite = torch.isfinite(output)
            assert expected is not None
            error = (output.float() - expected.float()).abs()
            print(
                "spark3 NaN probe: "
                f"{prefix}.b12x_native_vs_reference "
                f"max_abs={float(error.max().item())} "
                f"mean_abs={float(error.mean().item())} "
                f"native_max={float(output.float().abs().max().item())} "
                f"reference_max={float(expected.float().abs().max().item())}",
                flush=True,
            )
            if not bool(finite.all().item()):
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


def _wrap_indexed_cache_insert(owner: type[Any]) -> None:
    original = owner._insert_compressed_cache

    @functools.wraps(original)
    def checked(
        self: Any, latent: torch.Tensor | None, positions: torch.Tensor
    ) -> None:
        selected = (
            os.path.exists(_ARM_FILE)
            and getattr(self, "layer_id", None) == 2
            and latent is not None
        )
        if selected:
            from vllm.forward_context import get_forward_context

            _trace(f"{self.prefix}.compressed_latent", latent)
            metadata = get_forward_context().attn_metadata[
                self.compressor.k_cache_prefix
            ]
            slots = metadata.slot_mapping
            cache_positions = positions.div(
                self.compress_ratio, rounding_mode="floor"
            ).mul(self.compress_ratio)
            rotated, _ = self.rotary_emb(cache_positions, latent.clone().unsqueeze(-2))
            rotated = rotated.squeeze(-2)
            _trace(f"{self.prefix}.rotated_latent", rotated)
        original(self, latent, positions)
        if selected:
            from b12x.attention._shared.mla.compressed_reference import (
                _gather_cache_reference,
            )
            from vllm.models.deepseek_v41.nvidia.b12x_attention import (
                _flatten_cache,
            )

            torch.cuda.synchronize(latent.device)
            valid = slots >= 0
            if bool(valid.any().item()):
                page_size = (
                    self._vllm_config.cache_config.block_size // self.compress_ratio
                )
                decoded, _ = _gather_cache_reference(
                    _flatten_cache(
                        self._compressed_kv_cache(),
                        name="DeepSeek V4.1 indexed cache",
                    ),
                    slots[valid],
                    page_size=page_size,
                    cache_format="deepseek_v41",
                    cache_kind="indexed",
                )
                expected = rotated[valid]
                error = (decoded.float() - expected.float()).abs()
                _trace(f"{self.prefix}.decoded_indexed_cache", decoded)
                print(
                    "spark3 NaN probe: "
                    f"{self.prefix}.indexed_cache_write "
                    f"slots={slots[valid].detach().cpu().tolist()} "
                    f"expected_max={float(expected.float().abs().max().item())} "
                    f"decoded_max={float(decoded.float().abs().max().item())} "
                    f"max_abs_error={float(error.max().item())} "
                    f"mean_abs_error={float(error.mean().item())}",
                    flush=True,
                )

    owner._insert_compressed_cache = checked


def _install() -> None:
    import vllm.models.deepseek_v41.nvidia.model as model
    from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention
    from vllm.models.deepseek_v41.nvidia.b12x_attention import (
        DeepseekV41B12xAttention,
    )
    from vllm.models.deepseek_v41.nvidia.engram import Engram
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    _wrap_model_state(DeepseekV41ModelState, DefaultModelState)
    _wrap_engram_prepare(Engram)
    _wrap_engram(Engram)
    _wrap_method(Engram, "embed", "embed")
    _wrap_b12x_wkv(B12xMxfp8LinearKernel)
    _wrap_indexed_cache_insert(DeepseekV41B12xAttention)
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
