#!/usr/bin/env python3
"""Repeat the fixed LRU request and retain complete outputs for review."""

import argparse
import ast
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


PROMPT = (
    "Write a short Python LRU cache class with get and put methods using "
    "OrderedDict. Give code and one sentence on complexity."
)


def assess(content):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
    if match is None:
        return "missing Python code block"
    try:
        tree = ast.parse(match.group(1))
    except SyntaxError as exc:
        return f"Python syntax error: {exc.msg} at line {exc.lineno}"
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for cls in classes:
        methods = {
            node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
        }
        if {"get", "put"} <= methods.keys():
            put_args = [arg.arg for arg in methods["put"].args.args]
            if put_args != ["self", "key", "value"]:
                return f"put signature uses {put_args!r}"
            return "pass"
    return "missing LRU class with get and put methods"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "chat_template_kwargs": {"thinking": False},
        "return_token_ids": True,
    }
    results = []
    import uuid
    for index in range(args.repeats):
        payload["cache_salt"] = uuid.uuid4().hex
        started = time.monotonic()
        request = Request(
            f"{args.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=180) as response:
            body = json.load(response)
        content = body["choices"][0]["message"]["content"] or ""
        verdict = assess(content)
        results.append(
            {
                "index": index,
                "elapsed_s": round(time.monotonic() - started, 3),
                "verdict": verdict,
                "content": content,
                "prompt_token_ids": body.get("prompt_token_ids"),
                "completion_token_ids": body["choices"][0].get("token_ids"),
                "usage": body.get("usage"),
            }
        )
        print(f"{index + 1}/{args.repeats}: {verdict}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"profile": args.profile, "request": payload, "runs": results}, indent=2)
        + "\n"
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
