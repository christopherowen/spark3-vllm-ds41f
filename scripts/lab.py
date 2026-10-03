#!/usr/bin/env python3
"""Experiment windows, runs and kernel-lab jobs on the three-Spark cluster.

usage (on dgx1, from the deployment checkout):
  scripts/lab.py window open [--minutes M] [--note TEXT]
  scripts/lab.py window status
  scripts/lab.py window close
  scripts/lab.py run SPEC.json [--dry-run] [--keep-open]
  scripts/lab.py queue add SPEC.json      (queue a spec for the open or next window)
  scripts/lab.py queue run [--idle-minutes M] [--minutes M]
  scripts/lab.py queue status
  scripts/lab.py kernel-local BUNDLE_DIR [--out DIR] [--dry-run]   (on any node, cluster stopped)
  scripts/lab.py watchdog [--max-age SECONDS] [--once]

A window holds the cluster for one experiment session. While it is open the hold file
(~/spark3-hold.json on dgx1) names this runner as holder; other agents must not stop,
restart, sync or benchmark the cluster, and this runner refuses to open a window over a
hold someone else wrote. Jobs inside a window run back to back with no restore of r5o in
between; the window closes (r5o booted, doctor --live, hold removed) when the run ends,
when a job fails, when someone writes ~/spark3-request.json, or at its time cap. A
watchdog started with the window closes it if the runner stops refreshing the heartbeat.

A queue runner opens one window and runs queued specs back to back (oldest first), so the
cluster keeps working while results are read and the next spec is written; it waits up to
--idle-minutes for a new spec before closing the window. Terminal lines for watchers:
"QUEUE job <name> done|failed", "QUEUE idle", "QUEUE closed". A "sync" job moves the node
checkouts to origin/main between jobs, so newly published arms can join an open window.

A run spec is JSON: {"experiment": "experiments/<dir>", "run": "rec6", "jobs": [...]} with
jobs of kind "measure" (arms booted in turn and measured; profile "lean" or "full";
"bracket" re-measures the first arm at the end), "validate" (a trace arm: boot A, scenario
traces, optional restart, per-node analysis in parallel while stopped) and "kernel" (a
kernel-lab bundle per node, run concurrently on the named nodes while the cluster is
stopped). Results keep the layout tables_arms.py reads.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
_spec = importlib.util.spec_from_loader("spark3", _loader)
spark3 = importlib.util.module_from_spec(_spec)
_loader.exec_module(spark3)

HOLD = Path.home() / "spark3-hold.json"
LAB_HOME = Path.home() / "spark3-lab"
REQUEST = Path.home() / "spark3-request.json"
HOLDER_PREFIX = "spark3-lab"
CONTAINER = "dsv41-karmic-kraken"
CANDIDATE = "/opt/spark3/candidate"
DEFAULT_MINUTES = 120
HEARTBEAT_MAX_AGE = 900
TRACE_LOG_DIR = "/cache/kkref/moe-checksums"

# Measurement profiles: (script, extra arguments, output stem). The bench command is built apart.
PROFILES = {
    "full": {
        "bench": ["--min-samples", "5", "--max-samples", "5", "--prefill-sizes", "1024,4096,16384,65536",
                  "--prefill-repeats", "3"],
        "extras": [("c1_distinct.py", ["--tokens", "256"], "c1-distinct"),
                   ("c8_distinct.py", ["--samples", "3", "--tokens", "256"], "c8-distinct"),
                   ("ttft_short.py", [], "ttft"),
                   ("mixed_latency.py", ["--rounds", "3"], "mixed")],
    },
    "lean": {
        "bench": ["--min-samples", "3", "--max-samples", "3", "--prefill-sizes", "1024,16384",
                  "--prefill-repeats", "2"],
        "extras": [("c1_distinct.py", ["--tokens", "256", "--prompts", "12"], "c1-distinct"),
                   ("c8_distinct.py", ["--samples", "2", "--tokens", "256"], "c8-distinct"),
                   ("ttft_short.py", ["--reps", "3"], "ttft"),
                   ("mixed_latency.py", ["--rounds", "3"], "mixed")],
    },
}
BENCH_BASE = ["--allow-mismatch", "--compare", "none", "--suites", "decode,prefill",
              "--decode-cases", "prose,json-nothink", "--concurrency", "1", "--prefill-text", "source"]
# Validation tiers: scenario sets for boot A and the restart (None = every scenario).
TIERS = {
    "full": {"scenarios": None, "restart_scenarios": None, "mixes": True},
    "quick": {"scenarios": "mixed,chunked,chunked_end,cache_long,distinct",
              "restart_scenarios": "cache_long,distinct", "mixes": False},
}


# ---------------------------------------------------------------- small helpers

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{now()} {message}", flush=True)


def parse_time(text: str) -> float:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def nodes_config() -> dict:
    return spark3.configuration()[1]


def head_url() -> str:
    head = next(node for node in nodes_config()["nodes"] if node["head"])
    return f"http://{head['management_ip']}:8000"


# ---------------------------------------------------------------- hold file and guards

def read_hold(path: Path = HOLD) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"holder": "unreadable hold file"}


def hold_is_ours(hold: dict | None) -> bool:
    return bool(hold) and str(hold.get("holder", "")).startswith(HOLDER_PREFIX)


def new_hold(minutes: int, note: str, at: float | None = None) -> dict:
    at = time.time() if at is None else at
    stamp = dt.datetime.fromtimestamp(at, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = dt.datetime.fromtimestamp(at + 60 * minutes, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "holder": f"{HOLDER_PREFIX}: {note}" if note else HOLDER_PREFIX,
        "window": f"{socket.gethostname()}-{int(at)}",
        "since": stamp,
        "expected_end": end,
        "heartbeat": stamp,
        "rule": "no GPU experiments, restarts, cluster sync or serving changes while this file exists",
        "request": f"to ask for the cluster, write {REQUEST}; the window closes after the current job",
    }


def heartbeat_age(hold: dict, at: float | None = None) -> float:
    at = time.time() if at is None else at
    return at - parse_time(hold["heartbeat"])


def beat(note: str | None = None) -> None:
    hold = read_hold()
    if not hold_is_ours(hold):
        raise SystemExit("the hold file is not ours; refusing to continue")
    hold["heartbeat"] = now()
    if note:
        hold["current"] = note
    write_json_atomic(HOLD, hold)


def window_should_close(hold: dict, at: float | None = None) -> str | None:
    at = time.time() if at is None else at
    if REQUEST.exists():
        return f"cluster requested ({REQUEST})"
    if at > parse_time(hold["expected_end"]):
        return "time cap reached"
    return None


def requests_in_flight(url: str | None = None) -> float | None:
    """Running plus waiting requests, or None when no server answers."""
    try:
        with urllib.request.urlopen((url or head_url()) + "/metrics", timeout=5) as response:
            text = response.read().decode()
    except OSError:
        return None
    total = 0.0
    for line in text.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            total += float(line.rsplit(" ", 1)[1])
    return total


def idle_for(seconds: int = 30, interval: int = 5) -> bool:
    deadline = time.time() + seconds
    while True:
        load = requests_in_flight()
        if load not in (None, 0.0):
            log(f"cluster busy ({load:g} requests)")
            return False
        if time.time() >= deadline:
            return True
        time.sleep(interval)


def published_problems() -> list[str]:
    problems = []
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=False)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                          capture_output=True).stdout.strip()
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "HEAD", "origin/main"]).returncode:
        problems.append(f"deployment commit {head[:8]} is not on origin/main")
    cluster, nodes, _ = spark3.configuration()
    repo = spark3.repository_path(cluster)
    for node in nodes["nodes"]:
        if node["head"]:
            continue
        theirs = spark3.run_ssh(nodes, node, "git", "-C", repo, "rev-parse", "HEAD").stdout.strip()
        if theirs != head:
            problems.append(f"{node['name']} checkout {theirs[:8]} differs from {head[:8]}")
    return problems


# ---------------------------------------------------------------- cluster actions

def spark3_cli(*arguments: str, dry: bool = False, capture: list | None = None) -> int:
    command = [sys.executable, str(ROOT / "bin" / "spark3"), *arguments]
    if dry:
        print("  $ " + shlex.join(command[1:]))
        return 0
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout.splitlines():
        if "docker run" not in line:
            print(line, flush=True)
    if capture is not None:
        capture.extend(process.stdout.splitlines())
    return process.returncode


def ready_seconds(lines: list[str]) -> float | None:
    """Seconds from launch to ready, from bin/spark3's 'cluster ready ... (+N.Ns)' line."""
    for line in lines:
        found = re.search(r"cluster ready.*\(\+([0-9.]+)s\)", line)
        if found:
            return float(found.group(1))
    return None


def boot(config: str, dry: bool = False) -> bool:
    # A lab window restores the promoted physical fabric. It cannot undo
    # recabling or safely clean up a candidate with a different node set.
    candidate_nodes = spark3.configuration(argparse.Namespace(cluster_config=config))[1]
    if candidate_nodes != nodes_config():
        log("lab windows require the promoted node topology; qualify a new fabric "
            "with explicit cluster commands before using it in lab runs")
        return False
    log(f"start {config}")
    started, lines = time.time(), []
    ok = spark3_cli("--cluster-config", config, "cluster", "start", "--replace", "--apply", dry=dry,
                    capture=lines) == 0
    if not dry:
        log(f"boot {Path(config).name}: ready +{ready_seconds(lines)} s, command {time.time() - started:.0f} s"
            f"{'' if ok else ' FAILED'}")
    return ok


def stop_cluster(dry: bool = False) -> bool:
    log("stop")
    return spark3_cli("cluster", "stop", "--remove", "--apply", "--parallel", dry=dry) == 0


def production_live() -> bool:
    process = subprocess.run([sys.executable, str(ROOT / "bin" / "spark3"), "doctor", "--live"], cwd=ROOT,
                             text=True, capture_output=True)
    return process.returncode == 0 and "live cluster matches" in process.stdout


def restore_production(dry: bool = False) -> bool:
    if not dry and production_live():
        log("r5o already serving")
        return True
    ok = boot(spark3.DEFAULT_CLUSTER_CONFIG, dry=dry)
    if not dry:
        ok = production_live() and ok
        log("doctor --live " + ("OK" if ok else "FAILED"))
    return ok


def container_running(node: dict | None = None) -> bool:
    command = ["docker", "ps", "-q", "--filter", f"name=^{CONTAINER}$"]
    if node is None:
        output = subprocess.run(command, text=True, capture_output=True).stdout
    else:
        output = spark3.run_ssh(nodes_config(), node, *command).stdout
    return bool(output.strip())


# ---------------------------------------------------------------- windows

def window_open(minutes: int, note: str, dry: bool = False) -> None:
    hold = read_hold()
    if hold and not hold_is_ours(hold):
        raise SystemExit(f"cluster held by {hold.get('holder')!r} since {hold.get('since')}; not opening")
    if hold_is_ours(hold):
        log(f"window {hold['window']} already open")
        return
    if dry:
        print(f"  would open a {minutes}-minute window after the publish and idle guards")
        return
    problems = published_problems()
    if problems:
        raise SystemExit("not opening: " + "; ".join(problems))
    if not idle_for(30):
        raise SystemExit("not opening: the cluster is serving requests")
    hold = new_hold(minutes, note)
    write_json_atomic(HOLD, hold)
    log(f"window {hold['window']} open until {hold['expected_end']}")
    watchdog_log = ROOT / "results" / "private" / "lab" / f"watchdog-{hold['window']}.log"
    watchdog_log.parent.mkdir(parents=True, exist_ok=True)
    with watchdog_log.open("a") as handle:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog"], cwd=ROOT,
                         stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)


def window_close(reason: str, dry: bool = False) -> bool:
    hold = read_hold()
    if hold and not hold_is_ours(hold):
        log(f"hold belongs to {hold.get('holder')!r}; leaving the cluster alone")
        return False
    log(f"closing window ({reason})")
    ok = restore_production(dry=dry)
    if not dry and ok and hold_is_ours(read_hold()):
        HOLD.unlink()
        log("window closed; hold removed")
    elif not ok:
        log("restore FAILED; hold kept so nobody assumes production is up")
    return ok


def watchdog(max_age: int, once: bool) -> None:
    log(f"watchdog: heartbeat limit {max_age} s")
    while True:
        hold = read_hold()
        if not hold_is_ours(hold):
            log("watchdog: no window of ours; exiting")
            return
        age = heartbeat_age(hold)
        if age > max_age:
            log(f"watchdog: heartbeat {age:.0f} s old; closing the window")
            window_close("watchdog: stale heartbeat")
            return
        if once:
            return
        time.sleep(60)


# ---------------------------------------------------------------- overlay checks

def image_file(image: str, path: str, cache: Path) -> Path | None:
    """The image's copy of PATH, extracted once with docker create/cp (no process is started)."""
    image_id = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True,
                              capture_output=True).stdout.strip().split(":")[-1][:16]
    target = cache / image_id / path.lstrip("/")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    created = subprocess.run(["docker", "create", image], text=True, capture_output=True)
    if created.returncode:
        return None
    container = created.stdout.strip()
    try:
        copied = subprocess.run(["docker", "cp", f"{container}:{path}", str(target)], capture_output=True)
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True)
    return target if copied.returncode == 0 else None


def fence_problems(mounts: list, image: str, home: str, read_image=None) -> list[str]:
    """Mounted B12X sources must not carry fewer TMA proxy fences than the image's own copy.

    An overlay built from an older B12X tree silently drops fences the image gained; the
    gemv-geom and mhc-mt overlays did exactly that before r5o.
    """
    read_image = read_image or (lambda path: image_file(image, path, Path.home() / ".cache" / "spark3-lab"))
    problems = []
    for source, destination, _mode in mounts:
        if not (destination.startswith(f"{CANDIDATE}/b12x/") and destination.endswith(".py")):
            continue
        local = Path(source.format(home=home))
        reference = read_image(destination)
        if reference is None or not local.is_file():
            continue
        mine = local.read_text(encoding="utf-8").count("fence_proxy")
        theirs = Path(reference).read_text(encoding="utf-8").count("fence_proxy")
        if mine < theirs:
            problems.append(f"{local}: {mine} fence_proxy calls, image has {theirs} ({destination})")
    return problems


def overlay_manifest(config: dict, home: str) -> list[str]:
    lines = []
    for source, destination, _mode in config["container"]["mounts"]:
        path = Path(source.format(home=home))
        if "spark3-overlay" in source and path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path} -> {destination}")
    return lines


# ---------------------------------------------------------------- plans

def arm_config_path(experiment: str, config: str) -> str:
    return config if "/" in config else f"{experiment}/{config}"


def measure_steps(spec: dict, job: dict) -> list[dict]:
    experiment, run = spec["experiment"], spec["run"]
    profile = dict(PROFILES[job.get("profile", "lean")])
    if "extras" in job:  # [[script, [args...], stem], ...]: the job's own workloads
        profile["extras"] = [tuple(extra) for extra in job["extras"]]
    arms = list(job["arms"])
    if job.get("bracket") and len(arms) > 1:
        first = dict(arms[0])
        first["label"] = f"{first['label']}-end"
        arms.append(first)
    steps = []
    for arm in arms:
        config, label = arm_config_path(experiment, arm["config"]), arm["label"]
        out = f"results/private/determinism/{run}"
        steps.append({"kind": "boot", "config": config, "label": label})
        steps.append({"kind": "curves", "label": label, "config": config, "out": f"{out}/curves-{label}.json"})
        if job.get("bench", True):
            steps.append({"kind": "cli", "label": label,
                          "argv": ["--cluster-config", config, "bench", *BENCH_BASE, *profile["bench"],
                                   "--output", f"results/private/bench/{run}-{label}"]})
        for script, extra, stem in profile["extras"]:
            steps.append({"kind": "script", "label": label,
                          "argv": [f"{experiment}/{script}", head_url(), *extra],
                          "out": f"{out}/{stem}-{label}.jsonl"})
    steps.append({"kind": "table", "argv": [f"{experiment}/tables_arms.py", run, *[a["label"] for a in arms]],
                  "out": f"results/private/lab/{run}-table.txt",
                  "curves": [f"results/private/determinism/{run}/curves-{a['label']}.json" for a in arms],
                  "summaries": [f"results/private/determinism/{run}/{stem}-{a['label']}.jsonl"
                                for _, _, stem in profile["extras"] for a in arms]})
    return steps


def validate_steps(spec: dict, job: dict) -> list[dict]:
    experiment = spec["experiment"]
    tier = TIERS[job.get("tier", "full")]
    config = arm_config_path(experiment, job["config"])
    out = f"results/private/determinism/{job['name']}"
    url = head_url()
    chunk = str(job.get("chunk", 4000))
    steps = [{"kind": "boot", "config": config, "label": job["name"]},
             {"kind": "fresh_inventory"},
             {"kind": "script", "argv": [f"{experiment}/c8_trace.py", url, f"{out}/c8", "--rounds", "1",
                                         "--tokens", "16"], "out": f"{out}/c8-runs.jsonl"}]
    scenarios = [f"{experiment}/scenario_trace.py", url, f"{out}/scenarios", "--repeats", "1", "--chunk", chunk]
    if tier["scenarios"]:
        scenarios += ["--scenarios", tier["scenarios"]]
    steps.append({"kind": "script", "argv": scenarios, "out": f"{out}/scenario-runs.jsonl"})
    dirs = ["c8", "scenarios"]
    if tier["mixes"]:
        steps.append({"kind": "script", "argv": [f"{experiment}/trace_mixes.py", url, f"{out}/mixes",
                                                 "--repeats", "1", "--tokens", "64", "--prompts",
                                                 "json,prose,long"], "out": f"{out}/mixes-runs.jsonl"})
        dirs.append("mixes")
    if job.get("restart", True):
        restart = [f"{experiment}/scenario_trace.py", url, f"{out}/scenarios", "--repeats", "1", "--chunk",
                   chunk, "--suffix", "-bootB"]
        if tier["restart_scenarios"]:
            restart += ["--scenarios", tier["restart_scenarios"]]
        steps.append({"kind": "boot", "config": config, "label": f"{job['name']} boot B"})
        steps.append({"kind": "script", "argv": restart, "out": f"{out}/scenario-runs-bootB.jsonl"})
    steps.append({"kind": "stop"})
    steps.append({"kind": "analyze", "out": out, "dirs": dirs, "config": config})
    return steps


PROFILE_WORKLOADS = {
    "decode": ("profile_decode.py", ["--tokens", "128"]),
    "c8": ("profile_c8.py", ["--case", "json", "--streams", "8", "--tokens", "128"]),
    "prefill": ("profile_prefill.py", ["--tokens", "16384"]),
}


def profile_steps(spec: dict, job: dict) -> list[dict]:
    """Torch-profiled arms (run55's method): per arm and workload one rank-0 trace, then kernel
    summaries and cost comparisons against the first arm, computed while the cluster is stopped."""
    experiment = spec["experiment"]
    out = f"results/private/determinism/{spec['run']}"
    workloads = job.get("workloads", list(PROFILE_WORKLOADS))
    steps = []
    for arm in job["arms"]:
        config = arm_config_path(experiment, arm["config"])
        steps.append({"kind": "boot", "config": config, "label": arm["label"]})
        steps.append({"kind": "curves", "label": arm["label"], "config": config,
                      "out": f"{out}/curves-{arm['label']}.json"})
        for workload in workloads:
            script, extra = PROFILE_WORKLOADS[workload]
            steps.append({"kind": "profile", "config": config, "label": arm["label"], "workload": workload,
                          "argv": [f"{experiment}/{script}", head_url(), *extra],
                          "out": f"{out}/{arm['label']}"})
    steps.append({"kind": "stop"})
    steps.append({"kind": "costs", "experiment": experiment, "out": out, "workloads": workloads,
                  "labels": [arm["label"] for arm in job["arms"]],
                  "config": arm_config_path(experiment, job["arms"][0]["config"])})
    return steps


def kernel_steps(spec: dict, job: dict) -> list[dict]:
    return [{"kind": "stop"},
            {"kind": "kernel", "bundles": job["bundles"], "out": f"results/private/lab/{spec['run']}"}]


def sync_steps(spec: dict, job: dict) -> list[dict]:
    return [{"kind": "sync"}]


def plan(spec: dict) -> list[dict]:
    builders = {"measure": measure_steps, "validate": validate_steps, "kernel": kernel_steps,
                "profile": profile_steps, "sync": sync_steps}
    steps = []
    for job in spec["jobs"]:
        if job["kind"] not in builders:
            raise SystemExit(f"unknown job kind {job['kind']!r}")
        steps.extend(builders[job["kind"]](spec, job))
    return steps


def describe_step(step: dict) -> str:
    if step["kind"] == "boot":
        return f"boot {step['config']} ({step['label']})"
    if step["kind"] == "cli":
        return "bin/spark3 " + shlex.join(step["argv"])
    if step["kind"] == "script":
        return "python3 " + shlex.join(step["argv"]) + f" > {step['out']}"
    if step["kind"] == "profile":
        return f"profile {step['label']} {step['workload']}: python3 " + shlex.join(step["argv"])
    if step["kind"] == "costs":
        return f"summarize kernels and compare costs against {step['labels'][0]} ({', '.join(step['workloads'])})"
    if step["kind"] == "sync":
        return "sync node checkouts to origin/main"
    if step["kind"] == "curves":
        return f"save the boot's measured step costs > {step['out']}"
    if step["kind"] == "table":
        return "python3 " + shlex.join(step["argv"]) + f" > {step['out']}"
    if step["kind"] == "analyze":
        return f"analyze {step['out']} {','.join(step['dirs'])} on every node, in parallel"
    if step["kind"] == "kernel":
        return "kernel-lab " + ", ".join(f"{b['bundle']}@{b['node']}" for b in step["bundles"])
    return step["kind"]


# ---------------------------------------------------------------- execution

def run_script(step: dict) -> int:
    out = ROOT / step["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        return subprocess.run([sys.executable, *step["argv"]], cwd=ROOT, stdout=handle,
                              stderr=subprocess.STDOUT).returncode


def fresh_inventory() -> None:
    nodes = nodes_config()
    for node in nodes["nodes"]:
        spark3.run_ssh(nodes, node, "docker", "exec", CONTAINER, "sh", "-c",
                       f"rm -f {TRACE_LOG_DIR}/inventory-* {TRACE_LOG_DIR}/plans-*")


def analyze(step: dict) -> dict:
    """analyze_trace4.py per node, the three nodes concurrently (cluster stopped: memory is free)."""
    if container_running():
        raise SystemExit("analysis needs the cluster stopped")
    config = json.loads((ROOT / step["config"]).read_text())
    image = config["container"]["image"]
    experiment = str(Path(step["config"]).parent)
    out = ROOT / step["out"]

    def one_node(node: str) -> None:
        for d in step["dirs"]:
            with (out / f"analysis4-{d}-{node}.jsonl").open("w") as handle:
                process = subprocess.run(
                    ["docker", "run", "--rm", "--memory=16g", "-e", "CUDA_VISIBLE_DEVICES=",
                     "-v", f"{ROOT / experiment / 'analyze_trace4.py'}:/a.py:ro", "-v", f"{out / d}:/t:ro",
                     "--entrypoint", "python3", image, "/a.py", "/t", "--node", node, "--chain", "12",
                     "--all-pairs"], text=True, capture_output=True)
                handle.write("".join(line + "\n" for line in (process.stdout + process.stderr).splitlines()
                                     if "Warn" not in line))

    names = [node["name"] for node in nodes_config()["nodes"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        list(pool.map(one_node, names))
    summary = summarize_analysis(out)
    write_json_atomic(out / "analysis-summary.json", summary)
    return summary


def profiler_dir(config_path: str) -> str:
    """The container's torch profiler directory, from the arm's --profiler-config."""
    args = json.loads((ROOT / config_path).read_text())["serve_args"]
    return json.loads(args[args.index("--profiler-config") + 1])["torch_profiler_dir"]


def profile_workload(step: dict) -> None:
    """One workload under the torch profiler. Every rank's trace is collected (rank r runs on
    node r): rank 0 alone hides time the others spend computing while it waits in collectives."""
    container_dir = profiler_dir(step["config"])
    host_relative = container_dir.replace("/cache/", "cache/", 1)
    nodes = nodes_config()
    repo = spark3.repository_path(spark3.configuration()[0])
    out = ROOT / step["out"]
    out.mkdir(parents=True, exist_ok=True)
    workload = step["workload"]

    def in_container(node: dict, script: str) -> subprocess.CompletedProcess:
        return spark3.run_ssh(nodes, node, "docker", "exec", CONTAINER, "sh", "-c", script)

    def each(function):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
            return list(pool.map(function, nodes["nodes"]))

    each(lambda node: in_container(node, f"mkdir -p {container_dir} && rm -rf {container_dir}/*.json.gz "
                                         f"{container_dir}/{workload}"))
    with (out / f"{workload}.json").open("w") as handle:
        code = subprocess.run([sys.executable, *step["argv"]], cwd=ROOT, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    log(f"profile {step['label']} {workload} exit {code}")

    def trace_bytes(node: dict) -> int:
        result = spark3.run_ssh(nodes, node, "sh", "-c",
                                f"cat {repo}/{host_relative}/*rank*.json.gz 2>/dev/null | wc -c")
        return int(result.stdout.strip() or 0)

    before = None
    for _ in range(120):  # traces are written after /stop_profile; wait until every node's stops growing
        sizes = each(trace_bytes)
        if all(sizes) and sizes == before:
            break
        before = sizes
        time.sleep(5)
    else:
        log(f"traces under {host_relative} did not settle: {before}")
    each(lambda node: in_container(node, f"mkdir -p {container_dir}/{workload} && "
                                         f"mv {container_dir}/*rank*.json.gz {container_dir}/{workload}/"))

    def fetch(node: dict) -> None:
        target = out / f"{workload}-trace" / node["name"]
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "-e", "ssh " + " ".join(spark3.ssh_options()),
                        f"{spark3.ssh_target(nodes, node)}:{repo}/{host_relative}/{workload}/", f"{target}/"],
                       check=False)

    each(fetch)


def profile_costs(step: dict) -> None:
    """summarize_kernels.py per arm, workload and rank (three at a time), then analyze_costs.py of
    each arm against the first, rank by rank; the cluster is stopped, so memory is free."""
    if container_running():
        raise SystemExit("kernel summaries need the cluster stopped")
    image = json.loads((ROOT / step["config"]).read_text())["container"]["image"]
    experiment, out = ROOT / step["experiment"], ROOT / step["out"]
    names = [node["name"] for node in nodes_config()["nodes"]]

    def docker_python(script: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "run", "--rm", "--memory=16g", "-e", "CUDA_VISIBLE_DEVICES=",
                               "-v", f"{experiment / script}:/s.py:ro", "-v", f"{out}:/o",
                               "--entrypoint", "python3", image, "/s.py", *args], text=True, capture_output=True)

    def summarize(item: tuple[str, str, str]) -> None:
        label, workload, node = item
        traces = sorted((out / label / f"{workload}-trace" / node).glob("*.json.gz"))
        if traces:
            docker_python("summarize_kernels.py", f"/o/{label}/{workload}-kernels-{node}.json",
                          f"/o/{label}/{workload}-trace/{node}/{traces[0].name}")

    items = [(label, workload, node) for label in step["labels"] for workload in step["workloads"]
             for node in names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(summarize, items))
    base = step["labels"][0]
    for label in step["labels"][1:]:
        for workload in step["workloads"]:
            for node in names:
                extra = ["--total"] if workload == "prefill" else []
                result = docker_python("analyze_costs.py", f"/o/{base}/{workload}-kernels-{node}.json",
                                       f"/o/{label}/{workload}-kernels-{node}.json", *extra, "--top", "25")
                (out / f"costs-{label}-{workload}-{node}.txt").write_text(result.stdout + result.stderr)
                log(f"costs {label} {workload} {node} against {base}:")
                print("\n".join(result.stdout.splitlines()[:12]), flush=True)


def sync_checkouts() -> bool:
    """Move every node's checkout to origin/main (between jobs: nothing runs from them)."""
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=False)
    moved = subprocess.run(["git", "-C", str(ROOT), "checkout", "-q", "--detach", "origin/main"])
    if moved.returncode:
        return False
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                          capture_output=True).stdout.strip()
    ok = spark3_cli("cluster", "sync", "--apply") == 0
    log(f"sync to {head} " + ("OK" if ok else "FAILED"))
    return ok


def summary_lines(paths: list[str]) -> str:
    """The last JSON line carrying a "summary" key of each workload output, grouped by workload."""
    lines = []
    for path in paths:
        file = ROOT / path
        if not file.exists():
            continue
        records = [l for l in file.read_text().splitlines() if l.startswith("{") and '"summary"' in l]
        lines.append(f"{file.stem}: {records[-1] if records else '(no summary)'}")
    return "\n".join(lines) + "\n"


CURVE_POINTS = (1, 6, 12, 24, 48, 256, 1024, 2048, 4096)


def save_curves(step: dict) -> None:
    """The boot's measured DSpark step costs (vllm-0048 log line), if the arm logs them.

    The profile runs on padding rows, which the MoE routers skip, unless the arm sets
    SPARK3_DSPARK_PROFILE_TOKENS=random (real rows, routed experts included); the file
    records which, since padded-row costs leave out a real step's largest part."""
    logs = subprocess.run(["docker", "logs", CONTAINER], text=True, capture_output=True)
    lines = [l for l in (logs.stdout + logs.stderr).splitlines() if "Profiled DSpark step costs" in l]
    if not lines:
        return
    found = re.search(r"verify (\[.*?\]\]) draft (\[.*\]\])", lines[-1])
    if not found:
        return
    out = ROOT / step["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = "padded"
    if step.get("config"):
        environment = json.loads((ROOT / step["config"]).read_text()).get("environment", {})
        if environment.get("SPARK3_DSPARK_PROFILE_TOKENS") == "random":
            rows = "real"
    write_json_atomic(out, {"rows": rows, "verify": json.loads(found.group(1)),
                            "draft": json.loads(found.group(2))})


def curves_table(paths: list[str]) -> str:
    """Verify step cost (ms) at fixed token counts per arm, against the first arm with the
    same row kind (padded rows skip routed MoE, so they never compare with real rows)."""
    rows, bases = [], {}
    for path in paths:
        file = ROOT / path
        if not file.exists():
            continue
        data = json.loads(file.read_text())
        kind = data.get("rows", "padded")
        verify = dict((int(x), float(y)) for x, y in data["verify"])
        label = file.stem.removeprefix("curves-")
        base = bases.get(kind)
        cells = []
        for point in CURVE_POINTS:
            nearest = min(verify, key=lambda tokens: abs(tokens - point)) if verify else None
            value = verify.get(nearest)
            if base is None or value is None or base.get(point) is None:
                cells.append(f"{nearest}:{value:.2f}" if value is not None else "-")
            else:
                cells.append(f"{nearest}:{value:.2f} ({100 * (value / base[point] - 1):+.1f}%)")
        if base is None:
            bases[kind] = {point: verify.get(min(verify, key=lambda t: abs(t - point))) for point in CURVE_POINTS}
        rows.append(f"  {label:<14} {kind:<6} " + "  ".join(cells))
    if not rows:
        return ""
    return ("target step cost at fixed token counts, ms (boot profile, vllm-0048; padded rows skip "
            "routed MoE, real rows include it):\n" + "\n".join(rows))


def summarize_analysis(out: Path) -> dict:
    totals = {}
    for path in sorted(out.glob("analysis4-*.jsonl")):
        entry = {"pairs": 0, "rows": 0, "differing": 0, "outputs_differ": 0, "cross_boot_pairs": 0,
                 "cross_boot_rows": 0, "recompute_mismatches": 0}
        for line in path.read_text().splitlines():
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if "rows_compared" in record:
                entry["pairs"] += 1
                entry["rows"] += record["rows_compared"]
                entry["differing"] += record["rows_differing"]
                entry["outputs_differ"] += not record["outputs_equal"]
                if ("bootB" in record["request"]) != ("bootB" in record["against"]):
                    entry["cross_boot_pairs"] += 1
                    entry["cross_boot_rows"] += record["rows_compared"]
            entry["recompute_mismatches"] += len(record.get("recomputed_mismatch") or [])
        totals[path.stem] = entry
    totals["passed"] = all(e["differing"] == 0 and e["outputs_differ"] == 0 and e["recompute_mismatches"] == 0
                           for k, e in totals.items() if k != "passed") and len(totals) > 0
    return totals


def run_kernel_bundles(step: dict) -> list[dict]:
    """Ship each bundle to ~/spark3-lab on its node and run kernel-local there, nodes concurrently.

    Bundles, their inputs and outputs stay outside the deployment checkout: a bundle under
    development never dirties a node's checkout (a dirty checkout blocks cluster start), and
    ignored files never enter a checkout by copy. Paths listed under "sync" (inputs such as
    captures, kept out of git) go to ~/spark3-lab/inputs/<path> on the other nodes.
    """
    nodes = nodes_config()
    cluster, _, _ = spark3.configuration()
    repo = spark3.repository_path(cluster)
    run = Path(step["out"]).name
    home = cluster["host"]["home"]

    def rsync(node: dict, source: Path, destination: str) -> None:
        spark3.run_ssh(nodes, node, "mkdir", "-p", str(Path(destination).parent))
        subprocess.run(["rsync", "-a", "--delete" if source.is_dir() else "--checksum",
                        "-e", "ssh " + " ".join(spark3.ssh_options()),
                        f"{source}/" if source.is_dir() else str(source),
                        f"{spark3.ssh_target(nodes, node)}:{destination}{'/' if source.is_dir() else ''}"],
                       check=False)

    def one(entry: dict) -> dict:
        node = spark3.node_by_name(nodes, entry["node"])
        bundle = ROOT / entry["bundle"]
        candidate = json.loads((bundle / "candidate.json").read_text())
        remote = f"{home}/spark3-lab/bundles/{run}/{bundle.name}"
        out = f"{home}/spark3-lab/results/{run}/{bundle.name}-{entry['node']}"
        rsync(node, bundle, remote)
        if not node["head"]:
            for relative in candidate.get("sync", []):
                rsync(node, ROOT / relative, f"{home}/spark3-lab/inputs/{relative}")
        process = spark3.run_ssh(nodes, node, "bash", "-lc",
                                 f"cd {shlex.quote(repo)} && python3 scripts/lab.py kernel-local "
                                 f"{shlex.quote(remote)} --out {shlex.quote(out)}")
        verdict = {"node": entry["node"], "bundle": entry["bundle"], "exit": process.returncode}
        fetched = spark3.run_ssh(nodes, node, "cat", f"{out}/verdict.json")
        if fetched.returncode == 0:
            verdict.update(json.loads(fetched.stdout))
        else:
            verdict.update({"passed": False, "errors": [process.stdout[-2000:], process.stderr[-2000:]]})
        return verdict

    # Same thermal starting point as a bench: every node below the cooling threshold.
    names = sorted({entry["node"] for entry in step["bundles"]})
    spark3.cool_nodes(nodes, [spark3.node_by_name(nodes, name) for name in names],
                      spark3.COOL_BELOW_C, spark3.COOL_TIMEOUT_S)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(step["bundles"])) as pool:
        verdicts = list(pool.map(one, step["bundles"]))
    local = ROOT / step["out"]
    local.mkdir(parents=True, exist_ok=True)
    write_json_atomic(local / "verdicts.json", {"verdicts": verdicts, "agree": verdicts_agree(verdicts)})
    return verdicts


def verdicts_agree(verdicts: list[dict]) -> bool:
    """The same bundle on different nodes must give the same row groups and bit-equality sets."""
    by_bundle = {}
    for verdict in verdicts:
        key = (tuple((g["op"], g["groups"]) for g in verdict.get("groups", [])),
               tuple((b["bits"], tuple(b.get("differs_at", []))) for b in verdict.get("bits", [])),
               tuple(verdict.get("failed", [])))
        by_bundle.setdefault(verdict["bundle"], set()).add(key)
    return all(len(keys) == 1 for keys in by_bundle.values())


def execute(spec: dict, dry: bool, keep_open: bool) -> int:
    steps = plan(spec)
    if dry:
        print(f"run {spec['run']}: {len(steps)} steps")
        for index, step in enumerate(steps, 1):
            print(f"{index:3d}. {describe_step(step)}")
        return 0
    hold = read_hold()
    if not hold_is_ours(hold):
        window_open(spec.get("minutes", DEFAULT_MINUTES), f"run {spec['run']}")
    home = spark3.configuration()[0]["host"]["home"]
    checked = set()
    failed = None
    current = {"note": "starting"}
    stop_beating = threading.Event()

    def heartbeat() -> None:
        # The heartbeat proves the runner is alive, not that a step is quick; the time cap
        # bounds a hung step, the watchdog a dead runner.
        while not stop_beating.wait(60):
            try:
                beat(current["note"])
            except SystemExit:
                return

    threading.Thread(target=heartbeat, daemon=True).start()
    for step in steps:
        hold = read_hold()
        if not hold_is_ours(hold):
            failed = "hold file lost"
            break
        reason = window_should_close(hold)
        if reason:
            failed = reason
            break
        current["note"] = describe_step(step)
        beat(current["note"])
        if step["kind"] == "boot":
            if step["config"] not in checked:
                config = json.loads((ROOT / step["config"]).read_text())
                problems = fence_problems(config["container"]["mounts"], config["container"]["image"], home)
                if problems:
                    failed = "stale overlay: " + "; ".join(problems)
                    break
                manifest = ROOT / "results" / "private" / "lab" / f"overlays-{spec['run']}.txt"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                with manifest.open("a") as handle:
                    handle.write(f"# {step['config']}\n" + "".join(l + "\n" for l in overlay_manifest(config, home)))
                checked.add(step["config"])
            if not boot(step["config"]):
                failed = f"boot of {step['config']} failed"
                break
        elif step["kind"] == "cli":
            log(f"bench {step['label']} exit {spark3_cli(*step['argv'])}")
        elif step["kind"] == "script":
            log(f"{Path(step['argv'][0]).name} {step.get('label', '')} exit {run_script(step)}")
        elif step["kind"] == "sync":
            if not sync_checkouts():
                failed = "sync failed"
                break
        elif step["kind"] == "profile":
            profile_workload(step)
        elif step["kind"] == "costs":
            profile_costs(step)
        elif step["kind"] == "curves":
            save_curves(step)
        elif step["kind"] == "table":
            if (ROOT / step["argv"][0]).exists():
                run_script(step)
                table = (ROOT / step["out"]).read_text()
            else:  # an experiment without tables_arms.py: each workload's summary line per arm
                table = summary_lines(step.get("summaries", []))
                (ROOT / step["out"]).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / step["out"]).write_text(table)
            curves = curves_table(step.get("curves", []))
            if curves:
                table += "\n" + curves + "\n"
                (ROOT / step["out"]).write_text(table)
            print(table, flush=True)
        elif step["kind"] == "fresh_inventory":
            fresh_inventory()
        elif step["kind"] == "stop":
            if container_running() and not stop_cluster():
                failed = "stop failed"
                break
        elif step["kind"] == "analyze":
            summary = analyze(step)
            log(f"analysed {step['out']}: {'PASSED' if summary['passed'] else 'DIFFERENCES'}")
        elif step["kind"] == "kernel":
            verdicts = run_kernel_bundles(step)
            for verdict in verdicts:
                log(f"kernel {verdict['bundle']}@{verdict['node']}: "
                    f"{'pass' if verdict.get('passed') else 'FAIL'} ({verdict.get('seconds', '?')} s)")
            log(f"kernel verdicts {'agree' if verdicts_agree(verdicts) else 'DISAGREE'} across nodes")
    stop_beating.set()
    if failed:
        log(f"run stopped: {failed}")
    if failed or not keep_open:
        window_close(failed or "run complete")
    log("done")
    return 1 if failed else 0


# ---------------------------------------------------------------- queue

QUEUE = ROOT / "results" / "private" / "lab" / "queue"


def queue_add(spec_path: str) -> Path:
    spec = json.loads(Path(spec_path).read_text())
    plan(spec)  # reject a malformed spec now, not when its turn comes
    QUEUE.mkdir(parents=True, exist_ok=True)
    target = QUEUE / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{Path(spec_path).name}"
    target.write_text(json.dumps(spec, indent=2) + "\n")
    log(f"QUEUE added {target.name}")
    return target


def queued() -> list[Path]:
    return sorted(QUEUE.glob("*.json")) if QUEUE.is_dir() else []


def queue_run(idle_minutes: int, minutes: int) -> int:
    """Run queued specs back to back in one window; close it when the queue stays empty."""
    window_open(minutes, "queue")
    if not hold_is_ours(read_hold()):
        return 1
    me = Path(__file__).resolve()
    code_at_start = hashlib.sha256(me.read_bytes()).hexdigest()
    for state in ("running", "done", "failed"):
        (QUEUE / state).mkdir(parents=True, exist_ok=True)
    idle_since = None
    while True:
        hold = read_hold()
        if not hold_is_ours(hold):
            log("QUEUE closed (hold lost)")
            return 1
        reason = window_should_close(hold)
        if reason:
            window_close(reason)
            log(f"QUEUE closed ({reason})")
            return 0
        pending = queued()
        if not pending:
            if idle_since is None:
                idle_since = time.time()
                log(f"QUEUE idle (closing in {idle_minutes} min unless a spec arrives)")
            if time.time() - idle_since > 60 * idle_minutes:
                window_close("queue empty")
                log("QUEUE closed (empty)")
                return 0
            beat("queue idle")
            time.sleep(10)
            continue
        idle_since = None
        job = pending[0]
        running = QUEUE / "running" / job.name
        os.replace(job, running)
        log(f"QUEUE job {job.name} started")
        code = execute(json.loads(running.read_text()), dry=False, keep_open=True)
        os.replace(running, QUEUE / ("done" if code == 0 else "failed") / job.name)
        log(f"QUEUE job {job.name} {'done' if code == 0 else 'failed'}")
        if code:
            log("QUEUE closed (job failed; the window was closed by the run)")
            return code
        if hashlib.sha256(me.read_bytes()).hexdigest() != code_at_start:
            # A sync job brought new runner code: continue the same window on it.
            log("QUEUE restarting on the synced runner (window stays open)")
            os.execv(sys.executable, [sys.executable, str(me), "queue", "run",
                                      "--idle-minutes", str(idle_minutes), "--minutes", str(minutes)])


def queue_status() -> None:
    hold = read_hold()
    print("window:", (hold or {}).get("holder", "none"), (hold or {}).get("current", ""))
    for state in ("", "running", "done", "failed"):
        directory = QUEUE / state if state else QUEUE
        names = sorted(p.name for p in directory.glob("*.json")) if directory.is_dir() else []
        print(f"{state or 'pending'}: {', '.join(names) if names else '-'}")


# ---------------------------------------------------------------- kernel lab (node-local)

def compile_cache(candidate: dict) -> Path:
    """Per-node compile caches shared by every bundle; the GPU lock keeps jobs one at a time."""
    return Path(os.path.expanduser(candidate.get("cache", "~/.cache/spark3-lab/compile")))


def bundle_mounts(bundle: Path, candidate: dict) -> list[tuple[Path, str]]:
    """Mount sources: ~-paths and absolute paths as given; otherwise the bundle's own file, else a
    synced input under ~/spark3-lab/inputs, else the path in this node's checkout. A source that
    exists nowhere is reported by kernel-local; nothing falls back silently to an empty mount."""
    mounts = []
    for source, destination in candidate.get("mounts", []):
        resolved = Path(os.path.expanduser(source))
        if not resolved.is_absolute():
            for base in (bundle, LAB_HOME / "inputs", ROOT):
                if (base / source).exists():
                    resolved = base / source
                    break
            else:
                resolved = ROOT / source
        mounts.append((resolved, destination))
    return mounts


def bundle_command(bundle: Path, candidate: dict, out: Path) -> list[str]:
    """docker run for one kernel-lab bundle: overlay/{vllm,b12x}/... mounted at the candidate tree."""
    image = candidate["image"]
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host"]
    for key, value in sorted(candidate.get("env", {}).items()):
        command += ["-e", f"{key}={value}"]
    overlay = bundle / "overlay"
    if overlay.is_dir():
        for path in sorted(p for p in overlay.rglob("*") if p.is_file()):
            relative = path.relative_to(overlay)
            package = relative.parts[0]
            if package not in ("vllm", "b12x"):
                continue
            command += ["-v", f"{path}:{CANDIDATE}/{package}/{package}/{Path(*relative.parts[1:])}:ro"]
    for source, destination in bundle_mounts(bundle, candidate):
        command += ["-v", f"{source}:{destination}:ro"]
    command += ["-v", f"{compile_cache(candidate)}:/c"]
    command += ["-w", candidate.get("workdir", f"{CANDIDATE}/b12x"), "--entrypoint", "python3", image,
                *candidate["argv"]]
    return command


def suite_verdict(lines: list[str]) -> dict:
    """A test-suite bundle ("verdict": "exit", e.g. pytest): it passes when tests passed and none
    failed; kernel_local also requires exit status 0. Failed test ids are what nodes must agree on."""
    failed = sorted({line.split(" - ", 1)[0].split(" ", 1)[1].strip()
                     for line in lines if line.startswith(("FAILED ", "ERROR ")) and " " in line})
    summary = next((line.strip("= \n") for line in reversed(lines)
                    if re.search(r"\b\d+ (passed|failed|errors?|skipped)\b", line)), "")
    return {
        "passed": not failed and re.search(r"\b\d+ passed\b", summary) is not None
                  and re.search(r"\b\d+ (failed|errors?)\b", summary) is None,
        "summary": summary,
        "failed": failed,
        "batch_variant": [],
        "bits_unequal": [],
        "errors": failed[:20],
        "groups": [],
        "bits": [],
        "timings": [],
    }


def verdict_from_lines(lines: list[str], candidate: dict) -> dict:
    """Pass when every invariant configuration has one row group and every expected-equal pair matches."""
    if candidate.get("verdict") == "exit":
        return suite_verdict(lines)
    exempt = tuple(candidate.get("variant_configs", ["production", "ref2"]))
    groups, bits, timings, errors = [], [], [], []
    for line in lines:
        if "Traceback" in line or line.startswith(("RuntimeError", "ValueError", "AssertionError")):
            errors.append(line.strip())
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "groups" in record:
            groups.append(record)
        elif "bits" in record:
            bits.append(record)
        elif "timing_us" in record:
            timings.append(record)
        elif "error" in record:
            errors.append(f"{record.get('config')}: {record['error']}")
    variant = []
    for record in groups:
        name = record["op"].split()[1] if len(record["op"].split()) > 1 else record["op"]
        if record["groups"] != 1 and not name.startswith(exempt):
            variant.append(record["op"])
    unequal = [record["bits"] for record in bits if record.get("differs_at")]
    return {
        "passed": not (variant or unequal or errors) and bool(groups or bits),
        "batch_variant": variant,
        "bits_unequal": unequal,
        "errors": errors[:20],
        "groups": groups,
        "bits": bits,
        "timings": timings,
    }


def kernel_local(bundle_dir: str, out_dir: str | None, dry: bool) -> int:
    bundle = Path(bundle_dir).resolve()
    candidate = json.loads((bundle / "candidate.json").read_text())
    out = Path(out_dir).resolve() if out_dir else bundle / "out"
    command = bundle_command(bundle, candidate, out)
    if dry:
        print(shlex.join(command))
        return 0
    if container_running():
        raise SystemExit(f"{CONTAINER} is running on this node; the kernel lab needs the GPU to itself")
    missing = [str(source) for source, _ in bundle_mounts(bundle, candidate) if not source.exists()]
    if missing:
        raise SystemExit("missing bundle inputs: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    compile_cache(candidate).mkdir(parents=True, exist_ok=True)
    with open("/tmp/spark3-lab-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.time()
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
    (out / "output.txt").write_text(process.stdout)
    verdict = verdict_from_lines(process.stdout.splitlines(), candidate)
    image_id = subprocess.run(["docker", "image", "inspect", candidate["image"], "--format", "{{.Id}}"],
                              text=True, capture_output=True).stdout.strip()
    inputs = sorted(p for p in bundle.rglob("*") if p.is_file() and out not in p.parents)
    verdict.update({
        "node": socket.gethostname(), "image": image_id, "exit": process.returncode,
        "seconds": round(elapsed, 1),
        "inputs_sha256": {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    })
    if process.returncode:
        verdict["passed"] = False
    write_json_atomic(out / "verdict.json", verdict)
    log(f"kernel {bundle.name}: {'pass' if verdict['passed'] else 'FAIL'} in {elapsed:.0f} s")
    return 0 if verdict["passed"] else 1


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    window = commands.add_parser("window")
    window.add_argument("action", choices=("open", "status", "close"))
    window.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    window.add_argument("--note", default="")
    window.add_argument("--dry-run", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("spec")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-open", action="store_true", help="leave the window open after the run")
    queue = commands.add_parser("queue")
    queue.add_argument("action", choices=("add", "run", "status"))
    queue.add_argument("spec", nargs="?")
    queue.add_argument("--idle-minutes", type=int, default=10)
    queue.add_argument("--minutes", type=int, default=180)
    kernel = commands.add_parser("kernel-local")
    kernel.add_argument("bundle")
    kernel.add_argument("--out")
    kernel.add_argument("--dry-run", action="store_true")
    dog = commands.add_parser("watchdog")
    dog.add_argument("--max-age", type=int, default=HEARTBEAT_MAX_AGE)
    dog.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "window":
        if args.action == "open":
            window_open(args.minutes, args.note, dry=args.dry_run)
        elif args.action == "close":
            return 0 if window_close("closed by hand", dry=args.dry_run) else 1
        else:
            hold = read_hold()
            print(json.dumps(hold, indent=2) if hold else "no hold")
            if hold and "heartbeat" in hold:
                print(f"heartbeat age {heartbeat_age(hold):.0f} s")
        return 0
    if args.command == "run":
        spec = json.loads(Path(args.spec).read_text())
        return execute(spec, args.dry_run, args.keep_open)
    if args.command == "queue":
        if args.action == "add":
            if not args.spec:
                raise SystemExit("queue add needs a spec")
            queue_add(args.spec)
            return 0
        if args.action == "status":
            queue_status()
            return 0
        return queue_run(args.idle_minutes, args.minutes)
    if args.command == "kernel-local":
        return kernel_local(args.bundle, args.out, args.dry_run)
    watchdog(args.max_age, args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
