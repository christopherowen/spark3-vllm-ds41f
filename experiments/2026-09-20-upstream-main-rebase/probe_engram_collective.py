#!/usr/bin/env python3
"""Replay the TP3 disk-Engram all-reduce used by the candidate.

Launch one process on each Spark with ``torchrun``.  Disk Engram stages 24 hash
heads per token on every rank, with zeros for rows owned by another rank.  The
candidate sums those ``[5, 24, 256]`` BF16 buffers across TP ranks before WKV.
This probe compares the prepared RoCEnante reduction with NCCL byte-for-byte.
It allocates only small collective control buffers and tens of KiB of payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from b12x.preparation import PreparationSession, PreparedCall


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 3:
        raise ValueError(f"this diagnostic requires the TP3 world, got {world}")
    torch.cuda.set_device(0)
    # Match vLLM's communicator construction: metadata and B12X endpoint
    # exchange use a CPU/Gloo group, while the numerical oracle uses NCCL.
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    device_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=90))

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

    request = plan.request(name="diagnostic.engram.tp3-reduce", prepare_call=prepare)
    try:
        with PreparationSession(
            device=torch.device("cuda", 0),
            autotune=False,
            compile_workers=1,
        ) as session:
            session.prepare((request,))

            torch.manual_seed(20260923 + rank)
            local = torch.randn(
                5,
                24,
                256,
                dtype=torch.bfloat16,
                device="cuda",
            )
            token = torch.arange(5, device="cuda")[:, None, None]
            head = torch.arange(24, device="cuda")[None, :, None]
            local = torch.where((token * 24 + head) % world == rank, local, 0)
            expected = local.clone()
            dist.all_reduce(expected, group=device_group)
            actual = runtime.all_reduce(local, plan=plan)
            torch.cuda.synchronize()
            runtime.check_health()

            equal = bool(torch.equal(actual, expected))
            report = {
                "rank": rank,
                "local_shape": list(local.shape),
                "reduced_shape": list(actual.shape),
                "reduced_equal_nccl": equal,
                "reduced_finite": int(torch.isfinite(actual).sum().item()),
                "reduced_values": actual.numel(),
                "reduced_sha256": _sha256(actual),
                "runtime_stats": runtime.stats(),
            }
            print(json.dumps(report, sort_keys=True), flush=True)
            if not equal:
                raise RuntimeError(f"Engram all-reduce mismatch on rank {rank}")
            dist.barrier()
    finally:
        runtime.close()
        dist.destroy_process_group(device_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
