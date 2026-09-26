#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-26-dspark-policy/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
TAG = "vllm-ds41f-kkref:01f1b874c774-r4a"
VLLM_TREE = "0debe8533c8855a08432abfbfb4c95ec908afa22"
base = json.loads((ROOT / "config/cluster.json").read_text())


def set_json_arg(cfg: dict, flag: str, **changes) -> None:
    args = cfg["serve_args"]
    i = args.index(flag)
    value = json.loads(args[i + 1])
    value.update(changes)
    args[i + 1] = json.dumps(value, separators=(",", ":"))


def arm(env: dict | None = None, cache: str = "vllm-r4a", **spec) -> dict:
    cfg = copy.deepcopy(base)
    cfg["container"]["image"] = TAG
    cfg["container"]["expected_labels"]["local.spark3.vllm.tree"] = VLLM_TREE
    args = cfg["serve_args"]
    # Thinking is already the default when a request names neither key; the
    # explicit default overrode a client's enable_thinking=false.
    i = args.index("--default-chat-template-kwargs")
    del args[i:i + 2]
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = f"/cache/kkref/jit/{cache}"
    environment["SPARK3_DSPARK_PROFILE_REPLAYS"] = "15"
    environment["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/k3"
    environment.update(env or {})
    cfg["environment"] = dict(sorted(environment.items()))
    if spec:
        set_json_arg(cfg, "--speculative-config", **spec)
    return cfg


arms = {
    "base": arm(),
    "marginal": arm(
        {"SPARK3_DSPARK_VERIFY_RULE": "marginal"},
        adaptive_verification_cost_scale=1.0,
    ),
    "topk": arm(
        {"SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/k3-topk1024"},
        cache="vllm-r4a-topk",
        dspark_draft_topk=1024,
    ),
    "k5trace": arm(
        {
            "SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/k5",
            "SPARK3_DSPARK_TRACE": "/cache/kkref/dspark-trace/k5.jsonl",
        },
        cache="vllm-r4a-k5",
        num_speculative_tokens=5,
    ),
}
k5 = arms["k5trace"]["serve_args"]
k5[k5.index("--max-cudagraph-capture-size") + 1] = "48"
compilation = json.loads(k5[k5.index("--compilation-config") + 1])
compilation["cudagraph_capture_sizes"] += [40, 48]
k5[k5.index("--compilation-config") + 1] = json.dumps(compilation, separators=(",", ":"))

for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
