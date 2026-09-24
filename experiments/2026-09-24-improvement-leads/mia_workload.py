#!/usr/bin/env python3
"""MiaAI's overnight_bench workload (prompts, parameters, metric) against our API.

Reimplemented from benchmarks/overnight_bench.py at MiaAI-Lab 6b40a5e so no
third-party code runs: thinking off, 512 max tokens, C1 decode tok/s =
(completion_tokens - 1) / (last content delta - first content delta), C4 =
2 prose + 2 code concurrently, aggregate = tokens / wave wall time.
"""
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://10.0.1.71:8000"
PROSE = ("Write a detailed, flowing essay of about 800 words on the history and engineering "
         "of lighthouses, from the Pharos of Alexandria to automated LED beacons. Use full "
         "paragraphs only: no lists, no headings, no markdown.")
CODE = ("Write a complete Python module implementing a thread-safe LRU cache using a doubly "
        "linked list and a dict, with get, put, delete, resize and a stats() method, full "
        "docstrings and type hints, followed by a unittest suite covering eviction order, "
        "resizing and concurrent access. Output only the code.")
PROSE2 = ("Tell the story of a small fishing village over one year as the seasons change, following "
          "three families. Write it as continuous literary prose of about 800 words, no headings or lists.")
CHAT = ("Explain to a curious fifteen-year-old how vaccines teach the immune system to "
        "recognise a virus, including memory B cells and T cells, and why booster shots "
        "exist. Be warm and concrete.")
WORKLOADS = {
    "prose": dict(prompt=PROSE, temperature=0.0),
    "code": dict(prompt=CODE, temperature=0.0),
    "chat_sampled": dict(prompt=CHAT, temperature=0.7, top_p=0.95),
    "prose2": dict(prompt=PROSE2, temperature=0.0),
}


def one(workload, max_tokens):
    w = WORKLOADS[workload]
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": w["prompt"]}],
            "max_tokens": max_tokens, "temperature": w["temperature"], "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    if "top_p" in w:
        body["top_p"] = w["top_p"]
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first = last = usage = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if (delta.get("content") or "") + (delta.get("reasoning_content") or ""):
                    now = time.perf_counter()
                    first = first or now
                    last = now
    tokens = usage["completion_tokens"]
    return {"tokens": tokens, "ttft": first - t0,
            "decode_tps": (tokens - 1) / (last - first) if last > first else None}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for workload in WORKLOADS:
        one(workload, 64)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda w: one(w, 64), ["prose", "code", "prose", "code"]))
    results = {}
    for workload in WORKLOADS:
        runs = [one(workload, 512) for _ in range(reps)]
        values = [run["decode_tps"] for run in runs]
        results[f"c1-{workload}"] = {
            "median": round(statistics.median(values), 2), "mean": round(statistics.mean(values), 2),
            "min": round(min(values), 2), "max": round(max(values), 2),
            "tokens": [run["tokens"] for run in runs],
            "ttft_median": round(statistics.median(run["ttft"] for run in runs), 3)}
        print(workload, results[f"c1-{workload}"], flush=True)
    waves = []
    for _ in range(reps):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            runs = list(pool.map(lambda w: one(w, 512), ["prose", "code", "prose", "code"]))
        wall = time.perf_counter() - start
        waves.append({"agg": sum(run["tokens"] for run in runs) / wall,
                      "streams": [run["decode_tps"] for run in runs]})
    aggregates = [wave["agg"] for wave in waves]
    results["c4"] = {
        "agg_median": round(statistics.median(aggregates), 2), "agg_mean": round(statistics.mean(aggregates), 2),
        "agg_min": round(min(aggregates), 2), "agg_max": round(max(aggregates), 2),
        "per_stream_median": round(statistics.median(s for wave in waves for s in wave["streams"] if s), 2)}
    print("c4", results["c4"], flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
