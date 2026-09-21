#!/usr/bin/env python3
"""Exercise the real three-rank RoCEnante data path without loading a model.

Launch one process per node with ``torchrun`` and the same RoCE environment used
by the serving deployment.  The probe deliberately rendezvous immediately
before the first RDMA operation, compares the result with NCCL, and emits the
runtime's per-HCA counters so routing failures remain distinguishable from
vLLM startup failures.
"""

from __future__ import annotations

import json
import os
import traceback
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _log(event: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "rank": dist.get_rank() if dist.is_initialized() else None,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    from b12x.comm import roce

    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    runtime = None
    try:
        _log(
            "initializing",
            peer_hcas=os.environ.get("B12X_ROCE_PEER_HCAS"),
            hcas=os.environ.get("B12X_ROCE_HCA") or os.environ.get("NCCL_IB_HCA"),
            gid_index=os.environ.get("B12X_ROCE_GID_INDEX")
            or os.environ.get("NCCL_IB_GID_INDEX"),
        )
        runtime = roce.AllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            max_size=1 << 20,
            max_gather_bytes=4 << 20,
        )
        plan = None
        direct_prepared = None
        mode = os.environ.get("B12X_PROBE_MODE", "prepared-plan")
        if hasattr(roce, "plan") and mode == "prepared-plan":
            from b12x.comm.roce import _preparation
            from b12x.preparation import PreparationSession, PreparedCall

            query = roce.query_from_runtime(
                runtime,
                surface="AllReduce.all_reduce",
                call={"dtypes": ("float16", "bfloat16", "float32")},
                topology="roce_rdma",
                peer_hosts=tuple(
                    f"rank:{rank}" for rank in range(dist.get_world_size())
                ),
            )
            plan = roce.plan(query, runtime=runtime)
            seeds = [
                torch.zeros(16 // dtype.itemsize, dtype=dtype, device=device)
                for dtype in (torch.float16, torch.bfloat16, torch.float32)
            ]

            def prepare(state: object) -> PreparedCall:
                calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
                calls.append(_preparation.prepared_gather_call(state, inp=seeds[1]))

                def prime() -> list[object]:
                    dist.barrier()
                    _log("priming", api="prepared-plan", stats=runtime.stats())
                    return [call.run() for call in calls]

                return PreparedCall(
                    run=prime,
                    output=tuple(call.output for call in calls),
                    owners=(*seeds, *calls),
                )

            request = plan.request(
                name="rocenante.transport.probe", prepare_call=prepare
            )
            with PreparationSession(
                device=device,
                autotune=False,
                compile_workers=int(os.environ.get("B12X_PROBE_COMPILE_WORKERS", "2")),
            ) as session:
                session.prepare((request,))
        elif hasattr(runtime, "prepare"):
            runtime.prepare(
                (torch.float16, torch.bfloat16, torch.float32),
                padded_gather=True,
            )
            dist.barrier()
            _log("priming", api="legacy-eager", stats=runtime.stats())
            seed = torch.zeros(8, dtype=torch.bfloat16, device=device)
            runtime.all_reduce(seed)
        else:
            from b12x.comm.roce import _allgather_cute, _oneshot_cute

            dtypes = (torch.float16, torch.bfloat16, torch.float32)
            runtime._prepare_resources(dtypes, padded_gather=True)
            direct_prepared = SimpleNamespace(
                runtime=runtime,
                reduce_launchers={
                    dtype: _oneshot_cute.get_launcher(*runtime._launcher_key(dtype))
                    for dtype in dtypes
                },
                gather_launcher=_allgather_cute.get_launcher(
                    *runtime._gather_launcher_key()
                ),
            )
            dist.barrier()
            _log("priming", api="direct-prepared", stats=runtime.stats())
            seed = torch.zeros(8, dtype=torch.bfloat16, device=device)
            direct_scope = (
                torch.inference_mode()
                if os.environ.get("B12X_PROBE_INFERENCE_MODE") == "1"
                else nullcontext()
            )
            with direct_scope:
                runtime._run_prepared_all_reduce(
                    seed,
                    prepared=direct_prepared,
                )

        dist.barrier()
        for nbytes in (16, 4096, 256 * 1024, 1 << 20):
            numel = nbytes // torch.bfloat16.itemsize
            source = torch.arange(numel, dtype=torch.float32, device=device)
            source = (source + dist.get_rank()).to(torch.bfloat16)
            expected = source.clone()
            dist.all_reduce(expected)
            if plan is not None:
                actual = runtime.all_reduce(source, plan=plan)
            elif direct_prepared is not None:
                actual = runtime._run_prepared_all_reduce(
                    source,
                    prepared=direct_prepared,
                )
            else:
                actual = runtime.all_reduce(source)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=6e-2)
            dist.barrier()
            _log("size_passed", nbytes=nbytes, stats=runtime.stats())
        _log("passed", stats=runtime.stats())
    except Exception as exc:
        _log(
            "failed",
            error=repr(exc),
            stats=runtime.stats() if runtime is not None else None,
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
