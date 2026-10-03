"""Read-only layout report for the audited DS4.1 source/model combinations."""

import hashlib
import json
import math
from pathlib import Path

import topology


LAYOUT_PATH = "experiments/2026-10-03-transport-profiles/model-layout.json"


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def nvfp4_storage(rows: int, columns: int, alignment: dict) -> dict:
    """Serialized values and E4M3 scale storage, excluding allocator overhead."""
    if rows <= 0 or columns <= 0 or columns % alignment["nvfp4_scale_group"]:
        raise ValueError("NVFP4 dimensions must be positive with K divisible by the scale group")
    logical_scale_columns = columns // alignment["nvfp4_scale_group"]
    scale_rows = round_up(rows, alignment["nvfp4_scale_rows"])
    scale_columns = round_up(logical_scale_columns, alignment["nvfp4_scale_columns"])
    return {
        "logical_shape": [rows, columns],
        "packed_values_uint8_shape": [rows, columns // alignment["nvfp4_values_per_byte"]],
        "packed_values_bytes": rows * columns // alignment["nvfp4_values_per_byte"],
        "swizzled_scales_e4m3_storage_shape": [scale_rows, scale_columns],
        "scale_bytes": scale_rows * scale_columns,
        "scale_alignment_extra_bytes": scale_rows * scale_columns - rows * logical_scale_columns,
    }


def kernel_scratch(cluster: dict, expert: int, alignment: dict) -> dict:
    """Routed-expert scratch alignment of the selected kernel family (B12X unless named)."""
    if cluster.get("kernel_backend", "b12x") == "tilelang":
        return {
            "compact_moe_n64_path": False,
            "compact_moe_intermediate_width_when_selected": None,
            "note": "TileLang's routed-expert GEMMs read K in blocks of 128, or of 192 or 64 when 128 does not divide "
                    "the per-rank width (TP4's 576), so neither weights nor scratch are padded.",
        }
    compact = expert % 128 == 64
    return {
        "compact_moe_n64_path": compact,
        "compact_moe_intermediate_width_when_selected": round_up(expert, alignment["compact_moe_intermediate_channels"]) if compact else None,
        "note": "Scratch alignment is not expert weight padding; actual kernel selection also depends on routed rows.",
    }


def describe(root: Path, cluster: dict, capacity_bytes: int) -> dict:
    raw = (root / LAYOUT_PATH).read_bytes()
    audit = json.loads(raw)
    labels = cluster["container"]["expected_labels"]
    pair = {key: labels.get(f"local.spark3.{key}.tree") for key in ("vllm", "b12x")}
    if pair not in audit["source_pairs"]:
        raise ValueError("model layout needs a source audit for these vLLM/B12X trees")
    if not any(Path(mount[0]).name == audit["model_snapshot"] and mount[1] == "/models"
               for mount in cluster["container"]["mounts"]):
        raise ValueError("model layout audit does not match the mounted checkpoint")
    checkpoint_raw = (root / audit["checkpoint_config"]).read_bytes()
    if hashlib.sha256(checkpoint_raw).hexdigest() != audit["checkpoint_config_sha256"]:
        raise ValueError("checkpoint configuration receipt changed; re-audit model dimensions")
    for key, expected in audit["required_environment"].items():
        if cluster["environment"].get(key) != expected:
            raise ValueError(f"model layout audit requires {key}={expected}")
    if cluster["environment"].get("VLLM_DS41_BATCH_INVARIANT", "0") != "0":
        raise ValueError("batch-invariant execution needs its own model layout audit")
    tp = int(topology.argument(cluster, "--tensor-parallel-size"))
    checkpoint = json.loads(checkpoint_raw)["text_config"]
    dims = {
        "hidden_size": checkpoint["hidden_size"],
        "attention_heads": checkpoint["num_attention_heads"],
        "attention_output_groups": checkpoint["o_groups"],
        "engram_wkv_width": checkpoint["hidden_size"] * (checkpoint["hc_mult"] + 1),
        "vocabulary_rows": checkpoint["vocab_size"],
        "expert_intermediate_width": checkpoint["moe_intermediate_size"],
        "dspark_markov_rank": checkpoint["dspark_markov_rank"],
        "dspark_auxiliary_layers": len(checkpoint["dspark_target_layer_ids"]),
        "index_heads": checkpoint["index_n_heads"],
    }
    alignment = audit["alignment"]
    if tp not in (3, 4) or topology.argument(cluster, "--dtype") != "bfloat16":
        raise ValueError("model layout report covers the audited TP3/TP4 BF16 profiles")
    engram = json.loads(topology.argument(cluster, "--engram-config") or "{}")
    if engram.get("projection_tp") is not True:
        raise ValueError("model layout audit requires tensor-parallel Engram projection")

    def sharded(logical, padded):
        return {"logical_global": logical, "padded_global": padded,
                "extra_global": padded - logical, "allocated_per_rank": padded // tp}

    groups = round_up(dims["attention_output_groups"], tp)
    heads = groups * (dims["attention_heads"] // dims["attention_output_groups"])
    vocab = round_up(dims["vocabulary_rows"], math.lcm(alignment["global_vocabulary_rows"], tp))
    expert = dims["expert_intermediate_width"] // tp
    compile_config = json.loads(topology.argument(cluster, "--compilation-config"))
    graph_sizes = compile_config["cudagraph_capture_sizes"]
    draft = json.loads(topology.argument(cluster, "--speculative-config"))
    if draft.get("method") != "dspark" or draft.get("draft_tensor_parallel_size") != tp:
        raise ValueError("model layout audit requires DSpark with the target TP size")
    if not draft.get("enable_adaptive_verification") or compile_config.get("cudagraph_mode") != "FULL_AND_PIECEWISE":
        raise ValueError("model layout audit requires the audited adaptive-verification graph mode")
    if not graph_sizes or any(type(n) is not int or n <= 0 for n in graph_sizes) or graph_sizes != sorted(set(graph_sizes)):
        raise ValueError("graph capacities must be increasing positive integers")
    query_rows = 1 + draft["num_speculative_tokens"]
    max_requests = int(topology.argument(cluster, "--max-num-seqs"))
    graph_limit = int(topology.argument(cluster, "--max-cudagraph-capture-size"))
    chunk_limit = int(topology.argument(cluster, "--max-num-batched-tokens"))
    context_limit = min(max_requests * query_rows, graph_limit, chunk_limit)
    context_capacities = []
    bound = 1
    while bound < context_limit:
        context_capacities.append(bound)
        bound *= 2
    if context_limit > 0:
        context_capacities.append(context_limit)
    query_capacities = sorted({round_up(n, query_rows) for n in graph_sizes
                               if round_up(n, query_rows) <= min(max_requests * query_rows, graph_limit)})
    shard_rows = vocab // tp
    vocabulary_shards = [
        {"rank": rank, "first_token_id": rank * shard_rows,
         "real_rows": max(0, min(shard_rows, dims["vocabulary_rows"] - rank * shard_rows)),
         "allocated_rows": shard_rows,
         "padding_rows": max(0, (rank + 1) * shard_rows - dims["vocabulary_rows"])}
        for rank in range(tp)
    ]
    max_decode = max(int(topology.argument(cluster, "--max-cudagraph-capture-size")),
                     int(topology.argument(cluster, "--max-num-seqs")) * (1 + draft["num_speculative_tokens"]), tp)
    # sp_prefill.rows_above_bytes: the fewest live rows whose padded BF16
    # hidden-state buffer exceeds registered capacity, independent of dispatch.
    sp_min = max(max_decode + 1, round_up(capacity_bytes // (dims["hidden_size"] * 2) + 1, tp) - tp + 1)
    examples = []
    for rows in (1, 6, 19, 48, 205, 4096):
        sp_rows = round_up(rows, tp) if sp_min <= rows <= chunk_limit else None
        examples.append({"live_rows": rows,
                         "next_configured_graph_capacity": next((n for n in graph_sizes if n >= rows), None),
                         "next_draft_context_graph_capacity": next((n for n in context_capacities if n >= rows), None),
                         "prefill_sp_collective_rows_if_eligible": sp_rows,
                         "prefill_sp_extra_rows": sp_rows - rows if sp_rows is not None else None})
    return {
        "authority": "derived audit, not runtime options or a live observation",
        "audit_file": LAYOUT_PATH,
        "audit_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence": audit["evidence"],
        "model_snapshot": audit["model_snapshot"],
        "checkpoint_config_sha256": audit["checkpoint_config_sha256"],
        "source_trees": pair,
        "tensor_parallel_size": tp,
        "model_dimensions": {
            "attention_heads": sharded(dims["attention_heads"], heads),
            "attention_output_groups": sharded(dims["attention_output_groups"], groups),
            "engram_wkv_width": sharded(dims["engram_wkv_width"], round_up(dims["engram_wkv_width"], tp * alignment["engram_rows_per_rank"])),
            "target_vocabulary_rows": sharded(dims["vocabulary_rows"], vocab),
            "draft_aux_projection_output": sharded(dims["hidden_size"], round_up(dims["hidden_size"], tp * alignment["engram_rows_per_rank"])),
            "routed_expert_intermediate_width": sharded(dims["expert_intermediate_width"], dims["expert_intermediate_width"]),
        },
        "drafter": {
            "vocabulary_shards": vocabulary_shards,
            "vocabulary_note": "Target head, draft head and Markov output share these token partitions. Padded token logits are masked; scale-only rows are not token IDs.",
            "lm_head_nvfp4": nvfp4_storage(shard_rows, dims["hidden_size"], alignment),
            "markov_output_nvfp4": nvfp4_storage(shard_rows, dims["dspark_markov_rank"], alignment),
            "markov_input_embedding": {"replicated_shape": [dims["vocabulary_rows"], dims["dspark_markov_rank"]], "padding_rows": 0},
            "aux_projection_input_width": dims["hidden_size"] * dims["dspark_auxiliary_layers"],
            "aux_context_bf16_buffer_shape": [context_limit, dims["hidden_size"] * dims["dspark_auxiliary_layers"]],
            "quantized_activation_examples": [
                {"rows_passed_to_head": m,
                 "lm_head": nvfp4_storage(m, dims["hidden_size"], alignment),
                 "markov_output": nvfp4_storage(m, dims["dspark_markov_rank"], alignment)}
                for m in (1, 8, 48)
            ],
            "storage_note": "Serialized tensor storage only: excludes scalar scales, allocator rounding and plan-dependent GEMM workspaces. Activation examples use actual head input rows, not necessarily the scheduled batch rows.",
        },
        "kernel_scratch": kernel_scratch(cluster, expert, alignment),
        "scheduled_rows": {
            "target_graph_capacities": graph_sizes,
            "target_exact_low_concurrency_graphs": [
                {"requests": r, "rows": r * q}
                for r in range(1, min(2, max_requests) + 1)
                for q in range(1, query_rows + 1) if r * q <= graph_limit
            ],
            "draft_query_rows_per_request": query_rows,
            "draft_query_graph_capacities_if_full_supported": query_capacities,
            "draft_context_graph_capacities_if_full_supported": context_capacities,
            "prefill_sp_min_live_rows": sp_min,
            "prefill_chunk_limit": chunk_limit,
            "examples": examples,
            "note": "Configured graph ladder is not a live dispatch prediction: request counts, query lengths, captured descriptors and mixed/decode mode also matter. Exact low-concurrency target graphs supplement the ladder. Rows are per forward, not full prompt length. SP columns cover eligible encoder work before CED compaction; null means outside that path's range.",
        },
        "transport_layout": {
            "pack_bytes": alignment["rocenante_pack_bytes"],
            "all_reduce": "Eligible payload sizes are multiples of 16 bytes. Misaligned pointers use alignment scratch without changing payload size; unsupported sizes follow the explicit backend policy.",
            "all_gather": "Direct layout needs aligned input/output pointers and total bytes divisible by 16; last-dimension gathers also need row bytes divisible by 16. Other eligible shapes stage each entire shard rounded to 16 bytes, gather TP such shards and reshape; this does not round every logical row.",
        },
        "other": {
            "vision_weights": "replicated at TP3 and TP4 in the served SM121 implementation",
            "indexer": {"heads_per_rank": dims["index_heads"], "head_padding": 0, "note": "Indexer weights/heads are replicated; sequence-parallel row distribution is separate."},
            "determinism": "These profiles do not enable experimental batch-invariant LM-head row duplication.",
        },
    }
