# syntax=docker/dockerfile:1.7

# Canonical ARM64 nightly af1c0149, the same runtime base as the 2026-09-20
# weekend image. Local Inference Lab's karmic-kraken-beta vLLM sits seven
# commits after it with no native-source changes in between, so only the two
# stable CUDA extensions are rebuilt from the overlaid source.
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf

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

FROM ${BASE_IMAGE}

ARG VLLM_REVISION=unrecorded
ARG VLLM_TREE=unrecorded
ARG B12X_REVISION=unrecorded
ARG B12X_PATCH_HEAD=unrecorded
ARG B12X_TREE=unrecorded
ARG DEPLOYMENT_COMMIT=uncommitted

LABEL org.opencontainers.image.source="https://github.com/christopherowen/spark3-vllm-ds41f" \
      org.opencontainers.image.revision="${DEPLOYMENT_COMMIT}" \
      org.opencontainers.image.base.name="docker.io/vllm/vllm-openai" \
      org.opencontainers.image.base.digest="sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf" \
      local.spark3.vllm.upstream="https://github.com/local-inference-lab/vllm" \
      local.spark3.vllm.ref="integration/karmic-kraken-beta" \
      local.spark3.vllm.revision="${VLLM_REVISION}" \
      local.spark3.vllm.tree="${VLLM_TREE}" \
      local.spark3.b12x.upstream="https://github.com/local-inference-lab/b12x" \
      local.spark3.b12x.ref="integration/karmic-kraken-beta" \
      local.spark3.b12x.revision="${B12X_REVISION}" \
      local.spark3.b12x.patch-head="${B12X_PATCH_HEAD}" \
      local.spark3.b12x.tree="${B12X_TREE}"

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

# B12X builds its io_uring disk row reader at first use; it needs the header.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      liburing-dev liburing2 pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Keep install-only generated modules and matching native extensions from the
# base install without overwriting any candidate source file. Python modules
# that exist in the base install but were deleted from the candidate tree stay
# deleted; only generated or build-only files and directories are copied.
RUN python3 - <<'PY'
import shutil
from pathlib import Path

import vllm

installed = Path(vllm.__file__).parent
source = Path("/opt/spark3/candidate/vllm/vllm")
generated = {"_version.py", "third_party/flashmla/flash_mla_interface.py"}
copied, skipped = 0, []
for path in installed.rglob("*"):
    if path.is_dir() or "__pycache__" in path.parts:
        continue
    relative = path.relative_to(installed)
    target = source / relative
    if target.exists():
        continue
    if (
        path.suffix == ".py"
        and target.parent.is_dir()
        and relative.as_posix() not in generated
    ):
        skipped.append(relative.as_posix())
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    copied += 1
print(f"copied {copied} generated/native files; kept {len(skipped)} deletions")
for relative in sorted(skipped):
    print("  not restored:", relative)
PY

# B12X declares CuTe DSL 4.6.2 exactly, as did the weekend image. Install the
# hash-locked wheel set over the nightly's 4.7.1, then B12X without dependencies.
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
PY

ENV PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x

RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs python3 - <<'PY'
import importlib.metadata as metadata
import subprocess

assert metadata.version("nvidia-cutlass-dsl") == "4.6.2"
import b12x.attention.compressed_sparse_mla  # noqa: F401
import b12x.comm.roce  # noqa: F401
import b12x.integration.vllm.loader  # noqa: F401
import b12x.sequence.engram  # noqa: F401
from b12x.comm.roce._proxy import load
from b12x.loader._native import _build
import vllm.distributed.device_communicators.b12x_roce_all_reduce  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.dspark  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.model  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.vl_model  # noqa: F401
print("RoCE proxy ABI", load().roce_abi_version())
native_path = _build()
linked = subprocess.check_output(["ldd", native_path], text=True)
assert "liburing.so" in linked, "B12X was built without io_uring row staging"
for name in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
    with open(f"/opt/spark3/candidate/vllm/vllm/{name}.abi3.so", "rb") as handle:
        assert handle.read(4) == b"\x7fELF"
print("karmic-kraken reference imports passed")
PY
