#!/usr/bin/env python3
"""Generate the LRU answer one token per request, to separate decode-time
state from prefill. mode=salt: every step is a full prefill (unique
cache_salt). mode=cached: steps share the prefix cache (extend by one)."""
import ast, json, re, sys, time, uuid
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000"
mode, out = sys.argv[1], sys.argv[2]
chunk = int(mode.split(":")[1]) if mode.startswith("chunk:") else 1
receipt = json.load(open(sys.argv[3]))
prompt_ids = receipt["runs"][0]["prompt_token_ids"]

def post(path, payload):
    req = Request(URL + path, data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as r:
        return json.load(r)

ids, steps, eos = list(prompt_ids), [], {1}
salt_base = uuid.uuid4().hex
t0 = time.monotonic()
for step in range(200):
    payload = {"model": "deepseek-v4.1-flash", "prompt": ids, "max_tokens": chunk,
               "temperature": 0, "seed": 42, "logprobs": 3,
               "return_token_ids": True}
    if mode == "salt" or mode.startswith("chunk:"):
        payload["cache_salt"] = f"{salt_base}-{step}"
    body = post("/v1/completions", payload)
    ch = body["choices"][0]
    toks = ch.get("token_ids") or []
    steps.append({"step": step, "tokens": toks, "text": ch["text"]})
    ids.extend(toks)
    if not toks or any(t in eos for t in toks) or ch.get("finish_reason") == "stop" or len(ids) - len(prompt_ids) >= 200:
        break
gen = ids[len(prompt_ids):]
text = post("/detokenize", {"model": "deepseek-v4.1-flash", "tokens": gen})["prompt"]
m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
verdict = "missing code block"
if m:
    try:
        tree = ast.parse(m.group(1)); verdict = "parses"
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            if "put" in fns:
                args = [a.arg for a in fns["put"].args.args]
                verdict = "pass" if args == ["self", "key", "value"] else f"put args {args}"
    except SyntaxError as e:
        verdict = f"syntax error line {e.lineno}"
i = text.find("def put")
print(f"mode={mode} steps={len(gen)} {time.monotonic()-t0:.0f}s verdict={verdict} put={text[i:i+32]!r}")
json.dump({"mode": mode, "prompt_token_ids": prompt_ids, "generated": gen, "text": text,
           "verdict": verdict, "steps": steps}, open(out, "w"), indent=1)
