#!/usr/bin/env python3
"""Validate one prompt through the candidate B12X disk-Engram boundary.

This is an isolated, single-rank diagnostic.  It allocates only the bounded
hash/row staging buffers, never loads model weights, and compares every local
row selected for the prompt with a direct CPU read from the immutable
checkpoint.  Run it once for each TP rank inside the exact candidate image.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast


def _load_benchmark(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("b12x_ngram_ssd_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load B12X benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt", default="1 + 1 =")
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        default=pathlib.Path(
            "/opt/spark3/candidate/b12x/benchmarks/benchmark_ngram_ssd.py"
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.tp_rank < args.tp_size:
        parser.error("tp-rank must be within tp-size")

    torch.cuda.set_device(args.device)
    benchmark = _load_benchmark(args.benchmark)
    checkpoint = benchmark.Checkpoint(args.checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.checkpoint / "tokenizer.json")
    )
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    token_map, compressed = benchmark.build_compressed_token_map(tokenizer)
    config = checkpoint.config["text_config"]
    if compressed != config["engram_compressed_vocab_size"]:
        raise ValueError("tokenizer and checkpoint Engram geometry differ")

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
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        device=args.device,
        engram_resident_scales=False,
        engram_token_bound=True,
    )
    reports = []
    for owner in config["engram_layer_ids"]:
        case = benchmark.make_case(
            case_args,
            checkpoint,
            "engram",
            owner,
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
            checked_slices = correctness.pop("checked_slices")
            output = case.out[: len(token_ids)]
            finite = torch.isfinite(output)
            if not bool(finite.all().item()):
                raise RuntimeError(
                    f"layer {owner} disk Engram produced non-finite rows: "
                    f"nan={int(torch.isnan(output).sum().item())}, "
                    f"inf={int(torch.isinf(output).sum().item())}"
                )
            reports.append(
                {
                    "layer": owner,
                    "hash_ids_sha256": hashlib.sha256(
                        case.ids[: len(token_ids)].cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "finite_values": int(finite.sum().item()),
                    "total_values": output.numel(),
                    "reader_stats": case.table.stats(),
                    "correctness": {
                        **correctness,
                        "direct_checkpoint_rows_checked": len(checked_slices),
                    },
                }
            )
        finally:
            case.close()

    checkpoint.unchanged()
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "token_ids": token_ids,
                "tp_rank": args.tp_rank,
                "tp_size": args.tp_size,
                "layers": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
