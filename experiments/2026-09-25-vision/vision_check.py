#!/usr/bin/env python3
"""Ask about two images with known contents; every expected word must appear."""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.0.1.71:8000"
CHECKS = [
    ("shapes.png", "Describe the shapes and their colours in this image, and read the text in it exactly.",
     [("red",), ("circle",), ("blue",), ("square", "rectangle"), ("SPARK3",), ("42",)]),
    ("card.png", "What is the large title in this image, and what number is shown for Code aggregate?",
     [("DeepSeek",), ("173",)]),
]


def ask(image: Path, question: str) -> tuple[str, dict, float]:
    data = base64.b64encode(image.read_bytes()).decode()
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            {"type": "text", "text": question},
        ]}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    return reply["choices"][0]["message"]["content"] or "", reply.get("usage", {}), time.perf_counter() - started


failed = 0
for name, question, expected in CHECKS:
    answer, usage, seconds = ask(HERE / "fixtures" / name, question)
    missing = [" or ".join(options) for options in expected
               if not any(option.lower() in answer.lower() for option in options)]
    verdict = "pass" if not missing else f"FAIL (missing {', '.join(missing)})"
    failed += bool(missing)
    print(f"{name}: {verdict}; {usage.get('prompt_tokens')} prompt tokens, {seconds:.1f} s")
    print("  " + answer.strip().replace("\n", "\n  "))
sys.exit(1 if failed else 0)
