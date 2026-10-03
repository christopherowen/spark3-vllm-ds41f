"""The real DS4.1 vocabulary head shards: B12X's packed projection against cuBLAS BF16.

CPU-read the checkpoint's head.weight rows for one TP4 and one TP3 rank, pack
them with the image's B12X, check the exact round trip, then time each row
count under CUDA graphs with the default configuration and check accuracy
and batch invariance. Reports its checks like a test suite (a FAILED line per
failed check, then "N passed"), so the bundle uses the suite verdict, and exits
non-zero on any failed check.
"""
import json
import sys

import torch
from safetensors import safe_open

from b12x.gemm import packed_bf16_vocab_projection as projection
from b12x.preparation import PreparationSession, PreparedCall

SNAPSHOT = "/m/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
SHARDS = {"tp4-rank0": (0, 32320), "tp3-rank2": (2 * 43094, 129280)}
ROWS = (1, 6, 8, 16, 32, 48)


def graph_time(fn, calls=10, replays=20):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    return sorted(samples)[len(samples) // 2]


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    failures, report = [], {}
    with safe_open(f"{SNAPSHOT}/model-00043-of-00048.safetensors", "pt", device="cpu") as handle:
        full = handle.get_slice("head.weight")
        weights = {name: full[a:b].to(device) for name, (a, b) in SHARDS.items()}
    for name in list(weights):
        weight = weights.pop(name)
        n, k = weight.shape
        packed = projection.pack(weight)
        exact = torch.equal(projection.unpack(packed).view(torch.int16), weight.view(torch.int16))
        packed_bytes = sum(t.numel() * t.element_size() for t in (
            packed.values, packed.exception_rows, packed.exception_entries, packed.row_offsets))
        info = {"rows": n, "exponent_window": [packed.exponent_base, packed.exponent_base + 14],
                "exceptions": packed.exceptions, "round_trip_exact": exact,
                "bf16_bytes": weight.numel() * 2, "packed_bytes": packed_bytes}
        print(name, json.dumps(info), flush=True)
        failures += [] if exact else [f"{name}: round trip"]
        source = torch.randn(max(ROWS), k, device=device, dtype=torch.bfloat16)
        reference_full = source.float() @ weight.float().T
        results = {}
        with PreparationSession(device=device, autotune=False, compile_workers=4) as session:
            plans = {}
            requests = []
            for rows in ROWS:
                plans[rows] = projection.plan(projection.Caps(device=device, max_tokens=rows, in_features=k,
                                                              out_features=n))

                def call(state, rows=rows):
                    return PreparedCall(run=lambda: state.run(source[:rows], packed.values, packed.exception_rows,
                                                              packed.exception_entries, packed.row_offsets,
                                                              packed.exponent_base))

                requests.append(plans[rows].request(name=f"head.m{rows}", prepare_call=call))
            session.prepare(tuple(requests))
            session.freeze()
            alone = projection.run(projection.bind(plans[1], source=source[:1], weight=packed))
            for rows in ROWS:
                rows_source = source[:rows]
                binding = projection.bind(plans[rows], source=rows_source, weight=packed)
                output = projection.run(binding)
                cublas = torch.empty(rows, n, device=device, dtype=torch.bfloat16)
                t_cublas = graph_time(lambda: torch.matmul(rows_source, weight.T, out=cublas))
                with session.capture():
                    t_packed = graph_time(lambda: projection.run(binding))
                err_packed = (output.float() - reference_full[:rows]).abs().max().item()
                err_cublas = (cublas.float() - reference_full[:rows]).abs().max().item()
                invariant = torch.equal(output[:1].view(torch.int16), alone.view(torch.int16))
                results[rows] = {"cublas_us": round(t_cublas * 1e3, 1), "packed_us": round(t_packed * 1e3, 1),
                                 "packed_gbps": round(packed_bytes / t_packed / 1e6), "max_error_packed": err_packed,
                                 "max_error_cublas": err_cublas, "batch_invariant": invariant}
                print(f"  {name} rows={rows:2d}: cuBLAS {t_cublas * 1e3:7.1f} us, packed {t_packed * 1e3:7.1f} us "
                      f"({packed_bytes / t_packed / 1e6:.0f} GB/s, {t_cublas / t_packed:.2f}x); max error "
                      f"{err_packed:.3g} vs cuBLAS {err_cublas:.3g}; batch invariant {invariant}", flush=True)
                if not invariant:
                    failures.append(f"{name} rows={rows}: batch invariance")
                if err_packed > 2 * max(err_cublas, 1e-3):
                    failures.append(f"{name} rows={rows}: error {err_packed} vs cuBLAS {err_cublas}")
        report[name] = {"shard": info, "rows": results}
        del weight, packed
        torch.cuda.empty_cache()
    print(json.dumps({"report": report, "failures": failures}))
    checks = sum(1 + 2 * len(entry["rows"]) for entry in report.values())  # round trip; accuracy, invariance per row count
    for failure in failures:
        print(f"FAILED {failure.replace(' ', '_')} - check failed")
    summary = f"{checks - len(failures)} passed" + (f", {len(failures)} failed" if failures else "")
    print(f"===== {summary} =====")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
