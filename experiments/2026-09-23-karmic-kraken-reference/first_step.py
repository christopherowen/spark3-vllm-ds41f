#!/usr/bin/env python3
"""Compare decode-step logprobs (graph replay) with prefill logprobs (eager)."""
import json, sys, uuid
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000"
PROMPT = ("Write a short Python LRU cache class with get and put methods using "
          "OrderedDict. Give code and one sentence on complexity.")
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6

def post(path, payload):
    req = Request(URL + path, data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as r:
        return json.load(r)

a = post("/v1/chat/completions", {
    "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0, "max_tokens": steps, "logprobs": True, "top_logprobs": 5,
    "chat_template_kwargs": {"thinking": False}, "return_token_ids": True,
    "cache_salt": uuid.uuid4().hex})
prompt_ids = a["prompt_token_ids"]
gen = a["choices"][0]["token_ids"]
a_lp = a["choices"][0]["logprobs"]["content"]
print(f"prompt {len(prompt_ids)} tokens; generated {len(gen)}")
worst = 0.0
for k in range(len(gen)):
    b = post("/v1/completions", {
        "model": "deepseek-v4.1-flash", "prompt": prompt_ids + gen[:k], "max_tokens": 1,
        "temperature": 0, "logprobs": 5, "cache_salt": uuid.uuid4().hex})
    b_top = b["choices"][0]["logprobs"]["top_logprobs"][0]
    a_top = {e["token"]: e["logprob"] for e in a_lp[k]["top_logprobs"]}
    common = set(a_top) & set(b_top)
    diffs = {t: round(a_top[t] - b_top[t], 4) for t in common}
    mx = max((abs(v) for v in diffs.values()), default=float("nan"))
    worst = max(worst, mx if mx == mx else 0)
    kind = "prefill-logits" if k == 0 else f"decode step {k}"
    print(f"{kind:16} token={a_lp[k]['token']!r:14} top5-overlap={len(common)} "
          f"max|dlogprob|={mx:.4f}  A={sorted(a_top.items(), key=lambda x:-x[1])[:3]}  "
          f"B={sorted(b_top.items(), key=lambda x:-x[1])[:3]}")
