# syntax=docker/dockerfile:1.7

# Exact ARM64 manifest for the d05da62 nightly captured on 2026-09-20.
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:54c653f0a663ac04a92c25aba60f264c6eb203635b1ad1f627eb932ae453609b
FROM ${BASE_IMAGE}

ARG VLLM_BASE=d05da62e9ccdf8e342b15bf6785d83224cc165af
ARG VLLM_PATCH_HEAD=d6b40631cad4bf406f25093f56436f2b8ef10800
ARG VLLM_TREE=0b2a255f44e816178e8f3a2f0eb6ffbb9aca1576
ARG B12X_BASE=0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68
ARG B12X_PATCH_HEAD=de8e7fa971eb7ae4c21e634a8931407a0b404568
ARG B12X_TREE=4dbbf9f0eb784f379d0bc38db8195dd65088aaaa
ARG DEPLOYMENT_COMMIT=uncommitted

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

# The base image was built from the exact vLLM base above. Reuse only those
# matching native extensions, then expose the reviewable Python overlay.
RUN set -eux; \
    installed=$(python3 -c \
      'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)'); \
    cd "$installed"; \
    find . -type f \( -name '*.so' -o -name '_version.py' \) \
      -exec cp --parents -a '{}' /opt/spark3/candidate/vllm/vllm/ ';'

# Resolve B12X normally. Its candidate metadata now declares the same CUTLASS
# DSL 4.7.1 toolchain as this vLLM image; dependency conflicts are not hidden.
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
      /opt/spark3/candidate/b12x \
    && python3 -m pip check

ENV PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x

RUN python3 - <<'PY'
import importlib.metadata as metadata

assert metadata.version("nvidia-cutlass-dsl") == "4.7.1"
import b12x.attention.compressed_sparse_mla  # noqa: F401
import b12x.comm.roce  # noqa: F401
import vllm.models.deepseek_v41.nvidia.b12x_attention  # noqa: F401
print("candidate source imports and CUTLASS dependency gate passed")
PY
