#!/usr/bin/env python3
"""A/B the LRU gate across stream-synchronization points (runs on dgx1)."""
import ast, json, re, subprocess, sys, uuid
from urllib.request import Request, urlopen

CONTAINER = sys.argv[1]
REPEATS = int(sys.argv[2])
ARMS = json.loads(sys.argv[3])
OUT = sys.argv[4]
PEERS = [None, "swank@192.168.0.2", "swank@192.168.2.2"]
PROMPT = ("Write a short Python LRU cache class with get and put methods using "
          "OrderedDict. Give code and one sentence on complexity.")

def set_flag(cfg):
    script = ("rm -f /cache/kkref/trace/layer-trace.flag" if cfg is None else
              "mkdir -p /cache/kkref/trace && cat > /cache/kkref/trace/layer-trace.flag")
    data = None if cfg is None else json.dumps(cfg).encode()
    for peer in PEERS:
        cmd = ["docker", "exec", "-i", CONTAINER, "sh", "-c", script]
        if peer:
            cmd = ["ssh", "-o", "BatchMode=yes", peer, " ".join(
                ["docker", "exec", "-i", CONTAINER, "sh", "-c", repr(script)])]
        subprocess.run(cmd, input=data, check=True)

def verdict(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return "no code"
    try:
        tree = ast.parse(m.group(1))
    except SyntaxError as e:
        return f"syntax@{e.lineno}"
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        if "put" in fns:
            args = [a.arg for a in fns["put"].args.args]
            return "pass" if args == ["self", "key", "value"] else f"put:{args[-1]}"
    return "no put"

results = {}
for name, cfg in ARMS.items():
    set_flag(cfg)
    verdicts = []
    for _ in range(REPEATS):
        req = Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps({
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "seed": 42, "max_tokens": 200,
            "chat_template_kwargs": {"thinking": False},
            "cache_salt": uuid.uuid4().hex}).encode(),
            headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=900) as r:
            body = json.load(r)
        verdicts.append(verdict(body["choices"][0]["message"]["content"] or ""))
    set_flag(None)
    results[name] = verdicts
    print(f"{name:14} {sum(v == 'pass' for v in verdicts)}/{REPEATS}  {verdicts}", flush=True)
json.dump(results, open(OUT, "w"), indent=1)
