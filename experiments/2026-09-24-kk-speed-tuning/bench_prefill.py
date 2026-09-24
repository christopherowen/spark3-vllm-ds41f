#!/usr/bin/env python3
"""Cold prefill timing: one uncached prompt per size, one output token.

usage: bench_prefill.py --sizes 8192,32768,65536,131072 --output FILE
Each prompt is distinct filler text with a unique cache salt, so prefix
caching cannot serve it. Reports prompt tokens, wall time, and prefill tok/s.
"""
import argparse
import json
import time
import uuid
from urllib.request import Request, urlopen

MODEL = "deepseek-v4.1-flash"


def post(url, payload, timeout):
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def filler(tokens, seed):
    # Roughly one token per word for plain lowercase English words.
    words = ("river stone lantern orchard meadow harbor canyon glacier prairie "
             "summit valley forest desert island marsh delta ridge basin").split()
    body = " ".join(words[(i * 7 + seed) % len(words)] for i in range(tokens))
    return f"Document {seed}:\n{body}\n\nReply with the word ok."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--sizes", default="8192,32768,65536,131072")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = []
    for index, size in enumerate(int(v) for v in args.sizes.split(",")):
        prompt = filler(int(size * 0.97), seed=index + int(time.time()) % 1000)
        start = time.perf_counter()
        body = post(f"{args.url}/v1/completions", {
            "model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0,
            "cache_salt": uuid.uuid4().hex,
        }, timeout=1800)
        elapsed = time.perf_counter() - start
        prompt_tokens = body["usage"]["prompt_tokens"]
        results.append({"requested": size, "prompt_tokens": prompt_tokens,
                        "seconds": round(elapsed, 3),
                        "prefill_tps": round(prompt_tokens / elapsed, 1)})
        print(json.dumps(results[-1]), flush=True)
    with open(args.output, "w") as handle:
        json.dump({"sizes": results}, handle, indent=2)


if __name__ == "__main__":
    main()
