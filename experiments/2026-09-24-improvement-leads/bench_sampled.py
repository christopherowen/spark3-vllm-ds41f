#!/usr/bin/env python3
"""Sampled-traffic DSpark acceptance and throughput.

usage: bench_sampled.py --output FILE [--requests 16] [--concurrency 1,4]
Runs prose and code prompts at temperature 1.0 (top_p 0.95), 256 output
tokens, and reads vLLM's speculative-decoding counters from /metrics before
and after each phase, so acceptance is measured on exactly that traffic.
"""
import argparse
import concurrent.futures
import json
import re
import time
from urllib.request import Request, urlopen

MODEL = "deepseek-v4.1-flash"
PROMPTS = {
    "prose": "Write a short essay about how rivers shape the land and the people who live near them.",
    "code": ("Write a Python module with a thread-safe LRU cache class (get, put, "
             "delete, clear) and unit tests using unittest."),
}
COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total")


def metrics(url):
    text = urlopen(f"{url}/metrics", timeout=10).read().decode()
    values = {}
    for name in COUNTERS:
        match = re.search(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        values[name] = float(match.group(1)) if match else 0.0
    return values


def one(url, prompt, seed):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "seed": seed,
               "chat_template_kwargs": {"thinking": False}}
    request = Request(f"{url}/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        return json.load(response)["usage"]["completion_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = []
    for case, prompt in PROMPTS.items():
        for concurrency in (int(c) for c in args.concurrency.split(",")):
            before = metrics(args.url)
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                tokens = sum(pool.map(lambda i: one(args.url, prompt, 1000 + i),
                                      range(args.requests)))
            elapsed = time.perf_counter() - start
            after = metrics(args.url)
            delta = {k: after[k] - before[k] for k in COUNTERS}
            drafts = delta["vllm:spec_decode_num_drafts_total"] or 1
            row = {
                "case": case, "concurrency": concurrency, "requests": args.requests,
                "completion_tokens": tokens, "aggregate_tps": round(tokens / elapsed, 1),
                "accepted_per_draft": round(delta["vllm:spec_decode_num_accepted_tokens_total"] / drafts, 3),
                "acceptance_rate": round(delta["vllm:spec_decode_num_accepted_tokens_total"]
                                         / (delta["vllm:spec_decode_num_draft_tokens_total"] or 1), 3),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    with open(args.output, "w") as handle:
        json.dump({"sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 256},
                   "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
