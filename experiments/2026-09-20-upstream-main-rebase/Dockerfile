# syntax=docker/dockerfile:1.7

# Exact ARM64 manifest for the d05da62 nightly captured on 2026-09-20.
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:54c653f0a663ac04a92c25aba60f264c6eb203635b1ad1f627eb932ae453609b
FROM ${BASE_IMAGE}

ARG VLLM_BASE=d05da62e9ccdf8e342b15bf6785d83224cc165af
ARG VLLM_PATCH_HEAD=0fe6eed4b7b3a8bcc807d56cdd8a24021d13ea68
ARG VLLM_TREE=1b0bdd16150101ce3b817d06a1720e28789441ac
ARG B12X_BASE=0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68
ARG B12X_PATCH_HEAD=8099cee92c78dd2472749ca8003053ee73be908d
ARG B12X_TREE=a1523b067b4df679f9aa23d543fa324dc00770f0
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
print("candidate source imports and CUTLASS dependency gate passed")
PY
