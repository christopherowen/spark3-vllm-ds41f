#!/usr/bin/env python3
"""DS4.1 Flash TP3 routed-MoE decode timing: tiny decode vs the dynamic kernel.

Synthetic MXFP4 weights at the per-rank serving shape (E=384, K=5120,
I_tp=768, top-6, w31 layout, W4A8-MX, swiglu_limit=10), prepared and bound
through the same B12X calls vLLM uses (plan_weights, prepare_weights,
plan_execution with exact token-count variants, bind, run). Each timed CUDA
graph replays CALLS launches with different routes, so consecutive launches
read different experts, as consecutive layers do in serving.
B12X_W4A8_TINY_DECODE selects the kernel for 1-4 tokens.

usage: bench_ds41_moe_decode.py --m 1,2,3,4,8 --routing random,shared
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/opt/spark3/candidate/b12x")
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from benchmarks.moe_preparation import (  # noqa: E402
    prepared_call,
    request_for_capacity,
    scratch_for,
)
from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION  # noqa: E402
from b12x.preparation import FrozenMapping, PreparationSession  # noqa: E402

E, K, N, TOPK, LIMIT = 384, 5120, 768, 6, 10.0
CALLS = 8


def prepare_experts(weights):
    ones = torch.ones(E, dtype=torch.float32, device="cuda")
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
            w13_layout=fused_moe.W13Layout("w31"),
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=LIMIT,
            swiglu_alpha=None,
            swiglu_beta=None,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=E, hidden_size=K, intermediate_size=N
        ),
    )
    return fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.PackedWeights(
            w13=weights["w13_fp4"],
            w2=weights["w2_fp4"],
            w13_block_scales=weights["w13_mx"],
            w2_block_scales=weights["w2_mx"],
            w13_global_scales=ones,
            w2_global_scales=ones,
            input_scale=ones,
            intermediate_scale=ones,
            immutable_input_scales=True,
        ),
    )


def routes(m, pattern, gen):
    if pattern == "shared":
        ids = torch.randperm(E, generator=gen, device="cuda")[:TOPK]
        ids = ids.unsqueeze(0).expand(m, TOPK)
    else:
        logits = torch.randn(m, E, generator=gen, device="cuda")
        ids = torch.topk(logits, TOPK, dim=-1).indices
    weights = torch.softmax(torch.randn(m, TOPK, generator=gen, device="cuda"), -1)
    return ids.to(torch.int32).contiguous(), (weights * 1.5).float().contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="1,2,3,4,8")
    parser.add_argument("--routing", default="random,shared")
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    counts = tuple(int(v) for v in args.m.split(","))
    tiny = os.environ.get("B12X_W4A8_TINY_DECODE", "1")

    experts = prepare_experts(make_synthetic_mxfp4_moe(E, K, N, seed=7, device="cuda"))
    torch.cuda.empty_cache()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    inputs = {
        m: (
            (torch.randn(m, K, generator=gen, device="cuda") * 2.0).to(torch.bfloat16),
            *routes(m, "random", gen),
            torch.empty(m, K, dtype=torch.bfloat16, device="cuda"),
        )
        for m in counts
    }
    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=max(counts), top_k=TOPK, warmup_token_counts=counts,
        ),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
        invocation=FrozenMapping({"tuning_route_pattern": TUNING_WORKLOAD_VERSION}),
    )
    request = request_for_capacity(
        declaration,
        name="ds41-moe-decode",
        calls={
            m: prepared_call(
                output=inputs[m][3],
                bind=lambda state, scratch, m=m: state.bind(
                    scratch=scratch, a=inputs[m][0], experts=experts,
                    topk_ids=inputs[m][1], topk_weights=inputs[m][2],
                    output=inputs[m][3], input_scales_static=True,
                ),
            )
            for m in counts
        },
    )
    session = PreparationSession(device=torch.device("cuda"), autotune=False)
    result = session.prepare((request,))
    variants = getattr(declaration, "variants", None) or {}

    for pattern in args.routing.split(","):
        for m in counts:
            plan = variants.get(m, declaration)
            x, _, _, out = inputs[m]
            scratch = scratch_for(plan)
            bindings = []
            for _ in range(CALLS):
                ids, weights = routes(m, pattern, gen)
                bindings.append((ids, weights, fused_moe.bind(
                    plan, scratch=scratch, a=x, experts=experts, topk_ids=ids,
                    topk_weights=weights, output=out, input_scales_static=True,
                )))
            impl = ",".join(sorted({
                str(getattr(b, "implementation", "?")) for _, _, b in bindings
            }))

            def launch_all():
                for _, _, binding in bindings:
                    fused_moe.run(binding=binding)

            for _ in range(3):
                launch_all()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture():
                with torch.cuda.graph(graph):
                    launch_all()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000.0 / (args.replays * CALLS)
            print(f"tiny={tiny} routing={pattern:6s} m={m} impl={impl:8s} "
                  f"{us:7.1f} us/layer", flush=True)
            del graph, bindings
    torch.cuda.synchronize()
    result.close()
    session.close()


if __name__ == "__main__":
    main()
