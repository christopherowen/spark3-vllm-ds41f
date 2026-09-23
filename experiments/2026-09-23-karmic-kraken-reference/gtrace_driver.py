#!/usr/bin/env python3
"""Record one traced LRU generation. usage: gtrace_driver.py CONTAINER LABEL"""
import json, subprocess, sys, uuid
from urllib.request import Request, urlopen

container, label = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8000"
PROMPT = ("Write a short Python LRU cache class with get and put methods using "
          "OrderedDict. Give code and one sentence on complexity.")
d = f"/cache/kkref/trace/gtrace-{label}"

def sh(script, data=None):
    subprocess.run(["docker", "exec", "-i", container, "sh", "-c", script],
                   input=data, check=True)

def post(path, payload):
    req = Request(URL + path, data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as r:
        return json.load(r)

sh(f"rm -rf {d}; mkdir -p /cache/kkref/trace; printf %s {d} > /cache/kkref/trace/gtrace.flag")
try:
    body = post("/v1/chat/completions", {
        "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "max_tokens": 120, "chat_template_kwargs": {"thinking": False},
        "return_token_ids": True, "cache_salt": uuid.uuid4().hex})
    post("/v1/completions", {"model": "deepseek-v4.1-flash", "prompt": "ok",
                             "max_tokens": 1, "temperature": 0,
                             "cache_salt": uuid.uuid4().hex})
finally:
    sh("rm -f /cache/kkref/trace/gtrace.flag")
text = body["choices"][0]["message"]["content"]
record = {"prompt_ids": body["prompt_token_ids"], "generated": body["choices"][0]["token_ids"],
          "text": text}
sh(f"cat > {d}/trajectory.json", json.dumps(record).encode())
i = text.find("def put")
print(label, "generated", len(record["generated"]), repr(text[i:i + 30]))
