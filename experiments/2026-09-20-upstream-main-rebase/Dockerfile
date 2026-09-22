# syntax=docker/dockerfile:1.7

# Exact ARM64 manifest for the d05da62 nightly captured on 2026-09-20.
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:54c653f0a663ac04a92c25aba60f264c6eb203635b1ad1f627eb932ae453609b

FROM ${BASE_IMAGE} AS vllm-native-builder

USER root

COPY --from=vllm-source . /workspace/vllm/

# The runtime base intentionally supplies precompiled vLLM extensions. Patch
# 0024 changes one stable CUDA translation unit, so rebuild only that extension
# for GB10/SM121 instead of rebuilding or replacing the rest of the native
# stack. Keep compilation parallelism bounded; this stage runs on a host whose
# inference service is stopped, but it must not create a new memory hazard.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake \
    && rm -rf /var/lib/apt/lists/*
ENV TORCH_CUDA_ARCH_LIST=12.1a
RUN set -eux; \
    python_path=$(python3 -c 'import sys; print(":".join(sys.path))'); \
    cmake -S /workspace/vllm -B /tmp/vllm-native -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
      "-DVLLM_PYTHON_PATH=$python_path" \
      -DFETCHCONTENT_BASE_DIR=/workspace/vllm/.deps \
      -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
      -DCMAKE_JOB_POOLS:STRING=compile=2 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DNVCC_THREADS=1; \
    cmake --build /tmp/vllm-native -j=2 --target=_C_stable_libtorch; \
    cmake --install /tmp/vllm-native \
      --prefix /opt/vllm-native \
      --component _C_stable_libtorch; \
    test -f /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so

FROM ${BASE_IMAGE}

ARG VLLM_BASE=unrecorded
ARG VLLM_PATCH_HEAD=unrecorded
ARG VLLM_TREE=unrecorded
ARG B12X_BASE=unrecorded
ARG B12X_PATCH_HEAD=unrecorded
ARG B12X_TREE=unrecorded
ARG DEPLOYMENT_COMMIT=uncommitted
ARG BASE_NCCL_VERSION=2.30.7

LABEL org.opencontainers.image.source="https://github.com/christopherowen/spark3-vllm-ds41f" \
      org.opencontainers.image.revision="${DEPLOYMENT_COMMIT}" \
      org.opencontainers.image.base.name="docker.io/vllm/vllm-openai" \
      org.opencontainers.image.base.digest="sha256:54c653f0a663ac04a92c25aba60f264c6eb203635b1ad1f627eb932ae453609b" \
      local.spark3.vllm.upstream="https://github.com/vllm-project/vllm" \
      local.spark3.vllm.base="${VLLM_BASE}" \
      local.spark3.vllm.patch-head="${VLLM_PATCH_HEAD}" \
      local.spark3.vllm.tree="${VLLM_TREE}" \
      local.spark3.b12x.upstream="https://github.com/local-inference-lab/b12x" \
      local.spark3.b12x.base="${B12X_BASE}" \
      local.spark3.b12x.patch-head="${B12X_PATCH_HEAD}" \
      local.spark3.b12x.tree="${B12X_TREE}"

USER root

COPY --from=vllm-source . /opt/spark3/candidate/vllm/
COPY --from=b12x-source . /opt/spark3/candidate/b12x/
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so \
  /opt/spark3/candidate/vllm/vllm/_C_stable_libtorch.abi3.so

# B12X detects io_uring support while building its native loader. The runtime
# library alone is insufficient: compile the wheel with the development header
# and pkg-config metadata, then discard only apt's package indexes.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      liburing-dev liburing2 pkg-config \
    && rm -rf /var/lib/apt/lists/*

# The base image was built from the exact vLLM base above. Preserve install-only
# generated Python modules and matching native extensions that are absent from
# the Git tree, without overwriting any reviewable candidate source file.
RUN set -eux; \
    installed=$(python3 -c \
      'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)'); \
    cp -a --no-clobber "$installed"/. /opt/spark3/candidate/vllm/vllm/; \
    test -f \
      /opt/spark3/candidate/vllm/vllm/third_party/flashmla/flash_mla_interface.py

# The official nightly intentionally carries NCCL 2.30.7 over Torch's declared
# 2.29.7 dependency, and pip check also reports the ARM64 cuSPARSELt marker.
# Preserve that tested base-image state. Install B12X without allowing pip to
# downgrade the native stack, verify every applicable direct B12X requirement,
# and require the complete pip-check output to remain byte-for-byte unchanged.
RUN set -eux; \
    python3 -m pip check >/tmp/pip-check.base 2>&1 || true; \
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
        raise RuntimeError(
            f"B12X requires {requirement}, but {installed} is installed"
        )
PY
RUN set -eux; \
    test "$(python3 -c 'import importlib.metadata as m; print(m.version("nvidia-nccl-cu13"))')" \
      = "$BASE_NCCL_VERSION"; \
    python3 -m pip check >/tmp/pip-check.candidate 2>&1 || true; \
    diff -u /tmp/pip-check.base /tmp/pip-check.candidate

ENV PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x

RUN python3 - <<'PY'
import importlib.metadata as metadata
import subprocess

assert metadata.version("nvidia-cutlass-dsl") == "4.7.1"
import b12x.attention.compressed_sparse_mla  # noqa: F401
import b12x.comm.roce  # noqa: F401
import b12x.integration.vllm.loader  # noqa: F401
import b12x.sequence.engram  # noqa: F401
from b12x.loader._native import load
import vllm.models.deepseek_v41.nvidia.b12x_attention  # noqa: F401
import vllm.models.deepseek_v41.nvidia.engram  # noqa: F401
native = load()
linked = subprocess.check_output(["ldd", native.__file__], text=True)
assert "liburing.so" in linked, "B12X was built without io_uring row staging"
stable = "/opt/spark3/candidate/vllm/vllm/_C_stable_libtorch.abi3.so"
with open(stable, "rb") as handle:
    assert b"0, 8, 16, 24, 32, 64, 128" in handle.read(), (
        "rebuilt stable extension lacks the TP3 24-head dispatcher"
    )
print("candidate source imports and CUTLASS dependency gate passed")
PY
