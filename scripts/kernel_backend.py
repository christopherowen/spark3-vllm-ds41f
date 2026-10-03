"""Kernel backend policy: which library runs the DS4.1 model compute kernels.

B12X is the default and the promoted backend. TileLang selects the TileLang
compiler, DeepSeek's TileKernels and the DS4.1 TileLang kernels in the vLLM
patch stack. Collectives (RoCEnante, NCCL) and the B12X checkpoint loader are
outside this policy: both backends keep them.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath


DEFAULT = "b12x"
# Serve arguments and the environment value each backend requires.
BACKENDS = {
    "b12x": {
        "--attention-backend": "B12X",
        "--linear-backend": "b12x",
        "--moe-backend": "b12x",
    },
    "tilelang": {
        "--attention-backend": "TILELANG",
        "--linear-backend": "tilelang",
        "--moe-backend": "tilelang",
    },
}
ENVIRONMENT = "VLLM_DS41_KERNEL_BACKEND"
CACHE_ENVIRONMENT = "TILELANG_CACHE_DIR"
# The TileLang backend needs these optional build sources, this capability in
# the vLLM source manifest record, and an image built from them.
SOURCES = ("tilelang", "tile_kernels")
CAPABILITY = "tilelang-kernels"


def backend(cluster: dict) -> str:
    return cluster.get("kernel_backend", DEFAULT)


def label(name: str) -> str:
    return f"local.spark3.{name}.tree"


def argument(cluster: dict, flag: str) -> str | None:
    args = cluster.get("serve_args", [])
    if flag not in args:
        return None
    i = args.index(flag) + 1
    return args[i] if i < len(args) else None


def problems(cluster: dict) -> list[str]:
    """Serve arguments and environment that contradict the selected backend."""
    selected = backend(cluster)
    if selected not in BACKENDS:
        return [f"unknown kernel_backend {selected!r}; choose {' or '.join(BACKENDS)}"]
    errors = []
    for flag, wanted in BACKENDS[selected].items():
        value = argument(cluster, flag)
        if value != wanted:
            errors.append(f"kernel_backend {selected} requires {flag} {wanted}, got {value}")
    try:
        spec = json.loads(argument(cluster, "--speculative-config") or "{}")
    except ValueError:
        spec = None
    if isinstance(spec, dict) and spec:
        wanted = BACKENDS[selected]["--attention-backend"]
        if spec.get("attention_backend") != wanted:
            errors.append(
                f"kernel_backend {selected} requires speculative attention_backend {wanted}, "
                f"got {spec.get('attention_backend')}"
            )
    env = cluster.get("environment", {})
    # vLLM defaults to B12X when the variable is absent, so only TileLang
    # requires it; a value that is present must name the selected backend.
    value = env.get(ENVIRONMENT)
    if (selected != DEFAULT or value is not None) and value != selected:
        errors.append(f"kernel_backend {selected} requires {ENVIRONMENT}={selected}")
    if selected == "tilelang":
        cache = PurePosixPath(str(env.get(CACHE_ENVIRONMENT, "")))
        if not cache.is_absolute() or not cache.is_relative_to("/cache") or cache == PurePosixPath("/cache"):
            errors.append(f"kernel_backend tilelang requires {CACHE_ENVIRONMENT} under /cache/")
    return errors


def source_problems(cluster: dict, upstreams: dict, manifest: dict) -> list[str]:
    """The TileLang backend's build sources, vLLM capability and image trees."""
    if backend(cluster) != "tilelang":
        return []
    errors = []
    for name in SOURCES:
        if name not in upstreams.get("sources", {}):
            errors.append(f"kernel_backend tilelang requires source {name} in its upstreams lock")
    vllm = manifest.get("vllm", {})
    if CAPABILITY not in vllm.get("capabilities", []):
        errors.append(f"kernel_backend tilelang requires a vLLM source manifest with capability {CAPABILITY}")
    labels = cluster.get("container", {}).get("expected_labels", {})
    for name in ("vllm", *SOURCES):
        tree = manifest.get(name, {}).get("expected_tree")
        if tree is None or labels.get(label(name)) != tree:
            errors.append(
                f"kernel_backend tilelang requires container.expected_labels {label(name)} "
                "to match its source manifest tree"
            )
    return errors
