#!/usr/bin/env python3
"""Small deterministic OpenAI-compatible serving benchmark.

The script intentionally uses only the Python standard library so the same
client can run from the deployment checkout, a Spark, or an operator laptop.
It records every request separately and derives aggregate figures without
discarding failures.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


PROMPTS = {
    "prose": (
        "Explain, in precise technical prose, how speculative decoding can "
        "increase language-model serving throughput while preserving the "
        "target model's output distribution. Discuss acceptance rate, "
        "verification cost, batching, and latency."
    ),
    "code": (
        "Write a complete Python implementation of an LRU cache using only "
        "the standard library. Include type hints, O(1) get and put methods, "
        "capacity validation, and a short executable example. Explain the "
        "complexity after the code."
    ),
}


@dataclass
class RequestResult:
    request: int
    ok: bool
    status: int | None
    ttft_s: float | None
    elapsed_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    decode_tps_approx: float | None
    output_sha256: str
    error: str | None


def run_request(
    *,
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    request_index: int,
    timeout: float,
) -> RequestResult:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    status: int | None = None
    usage: dict[str, int] = {}
    output: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                pieces = [
                    delta.get("reasoning_content"),
                    delta.get("reasoning"),
                    delta.get("content"),
                ]
                emitted = "".join(piece for piece in pieces if isinstance(piece, str))
                if emitted:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output.append(emitted)
    except Exception as error:  # preserve every failed request in the receipt
        elapsed = time.perf_counter() - started
        return RequestResult(
            request=request_index,
            ok=False,
            status=status,
            ttft_s=None if first_token_at is None else first_token_at - started,
            elapsed_s=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            decode_tps_approx=None,
            output_sha256=hashlib.sha256("".join(output).encode()).hexdigest(),
            error=f"{type(error).__name__}: {error}",
        )

    ended = time.perf_counter()
    elapsed = ended - started
    ttft = None if first_token_at is None else first_token_at - started
    completion_tokens = usage.get("completion_tokens")
    decode_tps = None
    if completion_tokens and completion_tokens > 1 and ttft is not None and elapsed > ttft:
        decode_tps = (completion_tokens - 1) / (elapsed - ttft)
    return RequestResult(
        request=request_index,
        ok=status == 200 and first_token_at is not None,
        status=status,
        ttft_s=ttft,
        elapsed_s=elapsed,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        decode_tps_approx=decode_tps,
        output_sha256=hashlib.sha256("".join(output).encode()).hexdigest(),
        error=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://10.0.1.71:8000")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--case", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_request,
                url=endpoint,
                model=args.model,
                prompt=PROMPTS[args.case],
                output_tokens=args.output_tokens,
                request_index=index,
                timeout=args.timeout,
            )
            for index in range(args.concurrency)
        ]
        results = [future.result() for future in futures]
    wall_elapsed = time.perf_counter() - wall_started

    successful = [result for result in results if result.ok]
    completion_tokens = sum(result.completion_tokens or 0 for result in successful)
    summary = {
        "case": args.case,
        "concurrency": args.concurrency,
        "requested_output_tokens": args.output_tokens,
        "wall_elapsed_s": wall_elapsed,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "aggregate_e2e_tps": completion_tokens / wall_elapsed,
        "mean_ttft_s": (
            sum(result.ttft_s or 0 for result in successful) / len(successful)
            if successful
            else None
        ),
        "mean_decode_tps_approx": (
            sum(result.decode_tps_approx or 0 for result in successful)
            / len(successful)
            if successful
            else None
        ),
        "requests": [asdict(result) for result in results],
    }
    receipt = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(receipt, encoding="utf-8")
    print(receipt, end="")
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
