# syntax=docker/dockerfile:1.7

# Built by `bin/spark3 build image` from the contexts that `bin/spark3 build
# prepare` exports from upstreams.lock.json and patches/*/series.
#
# Canonical ARM64 nightly af1c0149, the same runtime base as the 2026-09-20
# weekend image. Local Inference Lab's karmic-kraken-beta vLLM sits seven
# commits after it with no native-source changes in between, so only the two
# stable CUDA extensions are rebuilt from the overlaid source.
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
# The image is the `runtime` stage unless the lock lists the TileLang sources;
# `bin/spark3 build image` then selects `runtime-tilelang`. BuildKit builds
# only the stages the selected one needs, so a B12X-only build neither builds
# TileLang nor needs its contexts, and its layers are unchanged.
ARG RUNTIME_STAGE=runtime

FROM ${BASE_IMAGE} AS vllm-native-builder

USER root

COPY --from=vllm-source . /workspace/vllm/
# CUTLASS headers come from a host-side checkout of the tag vLLM pins.
COPY --from=cutlass-source . /workspace/cutlass/
ENV VLLM_CUTLASS_SRC_DIR=/workspace/cutlass

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake \
      git \
    && rm -rf /var/lib/apt/lists/*
ENV TORCH_CUDA_ARCH_LIST=12.1a
ARG VLLM_COMPILE_JOBS=8
ARG VLLM_NVCC_THREADS=2
# Only the two stable extensions are rebuilt; the external projects (DeepGEMM,
# FlashMLA, flash-attention, Triton kernels, ...) produce other native modules
# that are taken prebuilt from the base image. Skip their configure-time
# clones: GitHub intermittently answers anonymous clones from this network with
# 401. A configure retry remains for any residual fetch.
RUN --mount=type=secret,id=github_token,required=false \
    set -eu; \
    if [ -s /run/secrets/github_token ]; then \
      github_auth=$(printf 'x-access-token:%s' "$(cat /run/secrets/github_token)" \
        | base64 | tr -d '\n'); \
      export GIT_CONFIG_COUNT=1; \
      export GIT_CONFIG_KEY_0=http.https://github.com/.extraheader; \
      export GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $github_auth"; \
    fi; \
    python_path=$(python3 -c 'import sys; print(":".join(sys.path))'); \
    sed -i -E 's#^([[:space:]]*)include\(cmake/external_projects/[a-z0-9_]+\.cmake\)#\1message(STATUS "spark3: external project skipped")#' \
      /workspace/vllm/CMakeLists.txt; \
    if grep -n 'include(cmake/external_projects/' /workspace/vllm/CMakeLists.txt; then exit 1; fi; \
    grep -q '#define CUTLASS_MAJOR 4' /workspace/cutlass/include/cutlass/version.h; \
    attempt=1; \
    until cmake -S /workspace/vllm -B /tmp/vllm-native -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
      "-DVLLM_PYTHON_PATH=$python_path" \
      -DFETCHCONTENT_BASE_DIR=/workspace/vllm/.deps \
      -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
      -DCMAKE_JOB_POOLS:STRING=compile=${VLLM_COMPILE_JOBS} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DNVCC_THREADS=${VLLM_NVCC_THREADS}; do \
      if [ "$attempt" -ge 8 ]; then exit 1; fi; \
      echo "cmake configure attempt $attempt failed; retrying in 45 s"; \
      attempt=$((attempt + 1)); sleep 45; \
    done; \
    cmake --build /tmp/vllm-native -j=${VLLM_COMPILE_JOBS} \
      --target _C_stable_libtorch _moe_C_stable_libtorch; \
    cmake --install /tmp/vllm-native --prefix /opt/vllm-native \
      --component _C_stable_libtorch; \
    cmake --install /tmp/vllm-native --prefix /opt/vllm-native \
      --component _moe_C_stable_libtorch; \
    test -f /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so; \
    test -f /opt/vllm-native/vllm/_moe_C_stable_libtorch.abi3.so

# NCCL 2.30.7 (the base image's nvidia-nccl-cu13 wheel), rebuilt from the
# pinned release tag with patches/nccl for SM121 only. The runtime stage
# replaces the wheel's libnccl.so.2 with it.
FROM ${BASE_IMAGE} AS nccl-builder

USER root

COPY --from=nccl-source . /workspace/nccl/
ARG NCCL_COMPILE_JOBS=4
RUN set -eux; \
    make -C /workspace/nccl -j${NCCL_COMPILE_JOBS} src.build \
      CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"; \
    mkdir -p /opt/nccl; \
    cp -L /workspace/nccl/build/lib/libnccl.so.2 /opt/nccl/libnccl.so.2; \
    sha256sum /opt/nccl/libnccl.so.2 | cut -d" " -f1 > /opt/nccl/libnccl.so.2.sha256

# TileLang 0.1.15 from tile-ai/tilelang with patches/tilelang and its pinned
# submodules, and DeepSeek's TileKernels, as wheels for the TileLang kernel
# backend. TileLang's CUDA stubs leave kernel compilation to run time, so no
# GPU architecture is fixed here. The isolated build environment is held to
# the base image's z3-solver, which TileLang links and loads at run time.
# TileKernels takes its version from setuptools-scm, which an export without
# Git metadata cannot supply.
FROM ${BASE_IMAGE} AS tilelang-builder

USER root

COPY --from=tilelang-source . /workspace/tilelang/
COPY --from=tile_kernels-source . /workspace/tile_kernels/
ARG TILELANG_COMPILE_JOBS=4
ARG TILE_KERNELS_VERSION=unrecorded
RUN set -eux; \
    z3=$(python3 -c 'import importlib.metadata as m; print(m.version("z3-solver"))'); \
    printf 'z3-solver==%s\n' "$z3" > /tmp/tilelang-build-constraints.txt; \
    export PIP_CONSTRAINT=/tmp/tilelang-build-constraints.txt; \
    export PIP_BUILD_CONSTRAINT=/tmp/tilelang-build-constraints.txt; \
    CMAKE_BUILD_PARALLEL_LEVEL=${TILELANG_COMPILE_JOBS} \
    CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
    NO_VERSION_LABEL=1 \
      python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/tilelang \
        /workspace/tilelang; \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TILE_KERNELS=${TILE_KERNELS_VERSION} \
      python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/tilelang \
        /workspace/tile_kernels; \
    ls /opt/tilelang/tilelang-*.whl /opt/tilelang/tile_kernels-*.whl

FROM ${BASE_IMAGE} AS runtime

ARG BASE_IMAGE_NAME=unrecorded
ARG BASE_IMAGE_DIGEST=unrecorded
ARG VLLM_UPSTREAM=unrecorded
ARG VLLM_REF=unrecorded
ARG VLLM_REVISION=unrecorded
ARG VLLM_PATCH_HEAD=unrecorded
ARG VLLM_TREE=unrecorded
ARG B12X_UPSTREAM=unrecorded
ARG B12X_REF=unrecorded
ARG B12X_REVISION=unrecorded
ARG B12X_PATCH_HEAD=unrecorded
ARG B12X_TREE=unrecorded
ARG NCCL_UPSTREAM=unrecorded
ARG NCCL_REF=unrecorded
ARG NCCL_REVISION=unrecorded
ARG NCCL_PATCH_HEAD=unrecorded
ARG NCCL_TREE=unrecorded
ARG DEPLOYMENT_COMMIT=uncommitted

LABEL org.opencontainers.image.source="https://github.com/christopherowen/spark3-vllm-ds41f" \
      org.opencontainers.image.revision="${DEPLOYMENT_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE_NAME}" \
      org.opencontainers.image.base.digest="${BASE_IMAGE_DIGEST}" \
      local.spark3.vllm.upstream="${VLLM_UPSTREAM}" \
      local.spark3.vllm.ref="${VLLM_REF}" \
      local.spark3.vllm.revision="${VLLM_REVISION}" \
      local.spark3.vllm.patch-head="${VLLM_PATCH_HEAD}" \
      local.spark3.vllm.tree="${VLLM_TREE}" \
      local.spark3.b12x.upstream="${B12X_UPSTREAM}" \
      local.spark3.b12x.ref="${B12X_REF}" \
      local.spark3.b12x.revision="${B12X_REVISION}" \
      local.spark3.b12x.patch-head="${B12X_PATCH_HEAD}" \
      local.spark3.b12x.tree="${B12X_TREE}" \
      local.spark3.nccl.upstream="${NCCL_UPSTREAM}" \
      local.spark3.nccl.ref="${NCCL_REF}" \
      local.spark3.nccl.revision="${NCCL_REVISION}" \
      local.spark3.nccl.patch-head="${NCCL_PATCH_HEAD}" \
      local.spark3.nccl.tree="${NCCL_TREE}"

USER root

COPY --from=vllm-source . /opt/spark3/candidate/vllm/
COPY --from=b12x-source . /opt/spark3/candidate/b12x/
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so \
  /opt/spark3/candidate/vllm/vllm/_C_stable_libtorch.abi3.so
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_moe_C_stable_libtorch.abi3.so \
  /opt/spark3/candidate/vllm/vllm/_moe_C_stable_libtorch.abi3.so
COPY --from=cutlass-dsl-wheels . /tmp/cutlass-dsl-wheels/
COPY --from=nccl-builder /opt/nccl/ /opt/spark3/nccl/
RUN set -eux; \
    target=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2; \
    test -f "$target"; \
    install -m 0755 /opt/spark3/nccl/libnccl.so.2 "$target"; \
    rm /opt/spark3/nccl/libnccl.so.2

# B12X builds its io_uring disk row reader at first use; it needs the header.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      liburing-dev liburing2 pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Keep install-only generated modules and matching native extensions from the
# base install without overwriting any candidate source file, then remove the
# files the candidate deliberately deletes relative to the base image's
# canonical commit (listed by `bin/spark3 build prepare` from git).
COPY --from=vllm-deletions deleted.txt /tmp/vllm-deleted.txt
RUN set -eux; \
    installed=$(python3 -c \
      'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)'); \
    cp -a --no-clobber "$installed"/. /opt/spark3/candidate/vllm/vllm/; \
    while read -r relative; do \
      [ -n "$relative" ] || continue; \
      rm -f "/opt/spark3/candidate/vllm/vllm/$relative"; \
      echo "kept deletion: $relative"; \
    done < /tmp/vllm-deleted.txt; \
    rm -f /tmp/vllm-deleted.txt; \
    test -f /opt/spark3/candidate/vllm/vllm/third_party/flashmla/flash_mla_interface.py

# The nightly base installs CuTe DSL 4.7.1, which vLLM and quack-kernels require;
# patches/b12x moves B12X's exact pin to it. Reinstall the hash-locked wheel set
# (the same version) and B12X without dependencies.
RUN set -eux; \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/cutlass-dsl-wheels/*.whl; \
    rm -rf /tmp/cutlass-dsl-wheels; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
      /opt/spark3/candidate/b12x
RUN python3 - <<'PY'
import importlib.metadata as metadata

from packaging.requirements import Requirement

for raw_requirement in metadata.requires("b12x") or ():
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = metadata.version(requirement.name)
    if requirement.specifier and installed not in requirement.specifier:
        raise RuntimeError(f"B12X requires {requirement}, but {installed} is installed")

# Every package built on CuTe DSL must accept the installed version: a locked
# wheel set older than one consumer's requirement silently breaks that consumer
# (quack-kernels 0.6.5 imports cutlass.base_dsl.enums, new in 4.7).
SHARED = ("nvidia-cutlass-dsl", "quack-kernels")
for consumer in ("vllm", "quack-kernels", "flashinfer-python", "nvidia-cudnn-frontend", "tokenspeed-mla"):
    try:
        requirements = metadata.requires(consumer) or ()
    except metadata.PackageNotFoundError:
        continue
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if not requirement.name.startswith(SHARED):
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        installed = metadata.version(requirement.name)
        if requirement.specifier and installed not in requirement.specifier:
            raise RuntimeError(f"{consumer} requires {requirement}, but {installed} is installed")
PY

ENV PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x

RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs python3 - <<'PY'
import importlib.metadata as metadata
import subprocess

assert metadata.version("nvidia-cutlass-dsl") == "4.7.1"
import cutlass.base_dsl.enums  # noqa: F401  (quack-kernels, the DS4.1 L2 prefetch)
import quack.compile_utils  # noqa: F401
import b12x.attention.compressed_sparse_mla  # noqa: F401
import b12x.comm.roce  # noqa: F401
import b12x.integration.vllm.loader  # noqa: F401
import b12x.sequence.engram  # noqa: F401
from b12x.comm.roce._proxy import load
from b12x.loader._native import _build
print("RoCE proxy ABI", load().roce_abi_version())
native_path = _build()
linked = subprocess.check_output(["ldd", native_path], text=True)
assert "liburing.so" in linked, "B12X was built without io_uring row staging"
import ctypes
import hashlib

# torch.cuda.nccl.version() reports the NCCL torch was compiled against, so
# ask the installed library itself (NCCL_VERSION_CODE 2.30.7 = 23007).
nccl = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
version = ctypes.c_int()
assert ctypes.CDLL(nccl).ncclGetVersion(ctypes.byref(version)) == 0
assert version.value == 23007, version.value
with open(nccl, "rb") as handle:
    built = open("/opt/spark3/nccl/libnccl.so.2.sha256").read().strip()
    assert hashlib.sha256(handle.read()).hexdigest() == built, "NCCL is not the patched build"
for name in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
    with open(f"/opt/spark3/candidate/vllm/vllm/{name}.abi3.so", "rb") as handle:
        assert handle.read(4) == b"\x7fELF"
print("B12X build checks passed")
PY

# The TileLang kernel backend: the patched TileLang replaces the base image's
# tilelang 0.1.12 (vLLM's pin), and TileKernels is added, both without
# dependencies. Every requirement must already hold; no shared package moves
# back to suit one consumer (docs/upstreams.md).
FROM runtime AS runtime-tilelang

ARG TILELANG_UPSTREAM=unrecorded
ARG TILELANG_REF=unrecorded
ARG TILELANG_REVISION=unrecorded
ARG TILELANG_PATCH_HEAD=unrecorded
ARG TILELANG_TREE=unrecorded
ARG TILELANG_VERSION=unrecorded
ARG TILE_KERNELS_UPSTREAM=unrecorded
ARG TILE_KERNELS_REF=unrecorded
ARG TILE_KERNELS_REVISION=unrecorded
ARG TILE_KERNELS_PATCH_HEAD=unrecorded
ARG TILE_KERNELS_TREE=unrecorded
ARG TILE_KERNELS_VERSION=unrecorded

LABEL local.spark3.tilelang.upstream="${TILELANG_UPSTREAM}" \
      local.spark3.tilelang.ref="${TILELANG_REF}" \
      local.spark3.tilelang.revision="${TILELANG_REVISION}" \
      local.spark3.tilelang.patch-head="${TILELANG_PATCH_HEAD}" \
      local.spark3.tilelang.tree="${TILELANG_TREE}" \
      local.spark3.tile_kernels.upstream="${TILE_KERNELS_UPSTREAM}" \
      local.spark3.tile_kernels.ref="${TILE_KERNELS_REF}" \
      local.spark3.tile_kernels.revision="${TILE_KERNELS_REVISION}" \
      local.spark3.tile_kernels.patch-head="${TILE_KERNELS_PATCH_HEAD}" \
      local.spark3.tile_kernels.tree="${TILE_KERNELS_TREE}"

RUN --mount=type=bind,from=tilelang-builder,source=/opt/tilelang,target=/tmp/tilelang-wheels \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/tilelang-wheels/tilelang-*.whl /tmp/tilelang-wheels/tile_kernels-*.whl
RUN --mount=type=bind,from=tilelang-builder,source=/opt/tilelang,target=/tmp/tilelang-wheels \
    TILELANG_VERSION=${TILELANG_VERSION} TILE_KERNELS_VERSION=${TILE_KERNELS_VERSION} \
    python3 - <<'PY'
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import os
import zipfile
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

built = {
    "tilelang": os.environ["TILELANG_VERSION"],
    "tile_kernels": os.environ["TILE_KERNELS_VERSION"],
}
for name, version in built.items():
    installed = metadata.version(name)
    assert installed == version, f"{name} {installed} is installed, expected {version}"
    copies = [d for d in metadata.distributions() if canonicalize_name(d.metadata["Name"] or "") == canonicalize_name(name)]
    assert len(copies) == 1, f"{len(copies)} {name} distributions are installed"
    # Every file of the wheel built above is installed unchanged.
    [wheel] = Path("/tmp/tilelang-wheels").glob(f"{name}-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        for path, digest, _ in csv.reader(io.TextIOWrapper(archive.open(record), "utf-8")):
            if not digest or ".dist-info/" in path or ".data/" in path:
                continue
            algorithm, wanted = digest.split("=", 1)
            data = Path(copies[0].locate_file(path)).read_bytes()
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest()).rstrip(b"=").decode()
            assert actual == wanted, f"{name}: installed {path} is not the built file"


def check(consumer: str, raw: str, shared: set[str] | None = None) -> None:
    try:
        requirement = Requirement(raw)
    except InvalidRequirement:
        return
    if shared is not None and canonicalize_name(requirement.name) not in shared:
        return
    if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
        return
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"{consumer} requires {requirement}, which is not installed") from None
    if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f"{consumer} requires {requirement}, but {installed} is installed")


for name in built:
    for raw in metadata.requires(name) or ():
        check(name, raw)

# TileLang and B12X's CuTe DSL stack share apache-tvm-ffi, and TileKernels
# needs TileLang: every consumer must accept the installed versions. The
# serving vLLM is the candidate source, so its requirements/cuda.txt pins
# apply rather than the base image's vLLM metadata.
SHARED = {"apache-tvm-ffi", "tilelang"}
for dist in metadata.distributions():
    consumer = canonicalize_name(dist.metadata["Name"] or "")
    if consumer != "vllm":
        for raw in dist.requires or ():
            check(consumer, raw, SHARED)
for line in Path("/opt/spark3/candidate/vllm/requirements/cuda.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if line and not line.startswith("-"):
        check("vllm (requirements/cuda.txt)", line, SHARED)
print("TileLang build checks passed")
PY

FROM ${RUNTIME_STAGE}
