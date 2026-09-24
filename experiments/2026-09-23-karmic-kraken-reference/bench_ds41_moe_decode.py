#!/usr/bin/env python3
"""DS4.1 Flash TP3 routed-MoE decode timing: tiny decode vs the dynamic kernel.

Adapted from B12X benchmarks/benchmark_ds4_moe.py. Synthetic MXFP4 weights at
the per-rank serving shape (E=384, K=5120, I_tp=768, top-6, w31 layout,
swiglu_limit=10), W4A8-MX. Each timed graph replays CALLS launches with
different routes, so consecutive launches read different experts as
consecutive layers do in serving. B12X_W4A8_TINY_DECODE selects the kernel.

usage: bench_ds41_moe_decode.py --m 1,2,3,4,8 --routing random,shared
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/opt/spark3/candidate/b12x/benchmarks")
from benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402

E, K, N, TOPK, LIMIT = 384, 5120, 768, 6, 10.0
CALLS = 8


def prepare(weights):
    from b12x.moe.fused_moe._impl import (
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )

    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=E,
        hidden_size=K,
        intermediate_size=N,
        w13_layout="w31",
    )
    return prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_global_scale=weights["alphas"],
        w2_global_scale=weights["alphas"],
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        a1_gscale=weights["input_scale"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )


def routes(m, pattern, gen, device):
    if pattern == "shared":
        ids = torch.randperm(E, generator=gen, device=device)[:TOPK]
        ids = ids.unsqueeze(0).expand(m, TOPK).contiguous()
    else:
        logits = torch.randn(m, E, generator=gen, device=device)
        ids = torch.topk(logits, TOPK, dim=-1).indices
    weights = torch.softmax(torch.randn(m, TOPK, generator=gen, device=device), -1)
    return ids.to(torch.int32).contiguous(), (weights * 1.5).float().contiguous()


def bench(experts, m, pattern, device, replays):
    from b12x.moe.fused_moe._impl import (
        allocate_tp_moe_workspace_pool,
        b12x_moe_fp4,
        build_tp_moe_fp4_binding,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    gen = torch.Generator(device=device)
    gen.manual_seed(1000 + m)
    x = (torch.randn(m, K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    out = torch.empty(m, K, dtype=torch.bfloat16, device=device)
    pool = allocate_tp_moe_workspace_pool()
    bindings = []
    for _ in range(CALLS):
        ids, weights = routes(m, pattern, gen, device)
        bindings.append((ids, weights, build_tp_moe_fp4_binding(
            scratch=pool, a=x, experts=experts, topk_weights=weights,
            topk_ids=ids, output=out, input_scales_static=True,
            quant_mode="w4a8_mx", swiglu_limit=LIMIT,
        )))
    impl = {b.implementation for _, _, b in bindings}

    def launch_all():
        for _, _, binding in bindings:
            b12x_moe_fp4(binding=binding)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            launch_all()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_all()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * CALLS), ",".join(sorted(impl))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="1,2,3,4,8")
    parser.add_argument("--routing", default="random,shared")
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    device = torch.device("cuda")
    tiny = os.environ.get("B12X_W4A8_TINY_DECODE", "1")
    weights = make_synthetic_mxfp4_moe(E, K, N, seed=7, device=device)
    experts = prepare(weights)
    del weights
    torch.cuda.empty_cache()
    for pattern in args.routing.split(","):
        for m in (int(v) for v in args.m.split(",")):
            us, impl = bench(experts, m, pattern, device, args.replays)
            print(f"tiny={tiny} routing={pattern:6s} m={m} impl={impl:8s} "
                  f"{us:7.1f} us/layer", flush=True)


if __name__ == "__main__":
    main()
