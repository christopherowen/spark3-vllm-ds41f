# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.config import ModelConfig, ParallelConfig, VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)


class _FP8DraftHead(nn.Module):
    """Block-32 FP8 copy of the shared bf16 LM head, used only for draft logits.

    The target keeps the bf16 head, so verification and every committed token
    are unchanged; only which tokens the draft proposes can differ. Scales are
    powers of two per 32x32 block (UE8M0), the checkpoint's own FP8 layout, so
    the copy runs on the same B12X block-32 GEMM as the model's FP8 linears.
    """

    _ROWS_PER_CHUNK = 2048

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        from vllm.models.deepseek_v4_1.b12x_layers import B12xFP8LinearMethod

        source = head.weight.detach()
        rows, cols = map(int, source.shape)
        if rows % 32 or cols % 32:
            raise ValueError(f"FP8 draft head needs 32-aligned shapes, got {(rows, cols)}")
        weight = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=source.device)
        scale = torch.empty(
            (rows // 32, cols // 32), dtype=torch.float8_e8m0fnu, device=source.device
        )
        # Chunked so the fp32 temporaries stay small on a memory-guarded host.
        for start in range(0, rows, self._ROWS_PER_CHUNK):
            end = min(start + self._ROWS_PER_CHUNK, rows)
            block = source[start:end].float().view(-1, 32, cols // 32, 32)
            amax = block.abs().amax(dim=(1, 3)).clamp_min(2.0**-120)
            factor = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
            quantized = (block / factor[:, None, :, None]).clamp(-448.0, 448.0)
            weight[start:end] = quantized.view(end - start, cols).to(torch.float8_e4m3fn)
            scale[start // 32 : end // 32] = factor.to(torch.float8_e8m0fnu)
            del block, amax, factor, quantized
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scale, requires_grad=False)
        self.tp_size = head.tp_size
        self.shard_indices = head.shard_indices
        self.quant_method = B12xFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
        self.quant_method.process_weights_after_loading(self)


def attach_fp8_draft_head(draft_model: nn.Module) -> None:
    """Route the draft's full-vocab logits through an FP8 copy of the head."""
    head = _FP8DraftHead(draft_model.lm_head)
    # A child module, so B12X preparation discovers and plans its GEMM.
    draft_model.fp8_draft_head = head
    norm = draft_model.model.norm
    processor = draft_model.logits_processor

    def compute_draft_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        return processor(head, norm(hidden_states))

    draft_model.compute_draft_logits = compute_draft_logits
    logger.info(
        "DSpark draft logits use a block-32 FP8 copy of the LM head %s",
        tuple(head.weight.shape),
    )


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def _get_dspark_parallel_config(
    parallel_config: ParallelConfig,
    tensor_parallel_size: int,
) -> ParallelConfig:
    if parallel_config.enable_eplb:
        logger.warning_once(
            "EPLB is disabled for the DSpark draft model. EPLB remains enabled "
            "for the target model."
        )

    return replace(
        parallel_config,
        pipeline_parallel_size=1,
        tensor_parallel_size=tensor_parallel_size,
        enable_eplb=False,
        eplb_config=replace(
            parallel_config.eplb_config,
            num_redundant_experts=0,
        ),
        enable_elastic_ep=False,
    )


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
        maybe_share_target_embed,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        parallel_config=_get_dspark_parallel_config(
            vllm_config.parallel_config,
            speculative_config.draft_parallel_config.tensor_parallel_size,
        ),
        kernel_config=(
            replace(
                vllm_config.kernel_config,
                moe_backend=speculative_config.moe_backend,
            )
            if speculative_config.moe_backend is not None
            else vllm_config.kernel_config
        ),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(vllm_config.load_config),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    if draft_model_config.get_vocab_size() <= target_vocab_size:
        maybe_share_target_embed(draft_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    if os.environ.get("SPARK3_DRAFT_HEAD_FP8") == "1":
        attach_fp8_draft_head(draft_model)

    return draft_model
