# syntax=docker/dockerfile:1.7

# This is the source-built successor to the captured node-local image. It is
# intentionally pinned by digest. See docker/README.md before promotion.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
FROM ${BASE_IMAGE} AS source-overlay

ARG VLLM_COMMIT=c687594b8a8082e18af9fe2f64eb5f9ee442e251
ARG B12X_COMMIT=f1c4e9dd5b1d841c10cda70ee49c3190cf3db0be
ARG B12X_UPSTREAM_PATCHES=02407f653609e2c4c49db2efb5f77835d2e04e32
ARG FLASHINFER_COMMIT=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1
ARG CUTLASS_COMMIT=da5e086dab31d63815acafdac9a9c5893b1c69e2
ARG DEPLOYMENT_COMMIT=uncommitted
ARG VLLM_PATCHSET_SHA256=unrecorded
ARG B12X_PATCHSET_SHA256=unrecorded

LABEL org.opencontainers.image.source="https://github.com/christopherowen/spark3-vllm-ds41f" \
      org.opencontainers.image.revision="${DEPLOYMENT_COMMIT}" \
      org.opencontainers.image.base.name="docker.io/vllm/vllm-openai" \
      org.opencontainers.image.base.digest="sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf" \
      local.spark3.vllm.upstream="https://github.com/vllm-project/vllm" \
      local.spark3.vllm.commit="${VLLM_COMMIT}" \
      local.spark3.vllm.patchset.sha256="${VLLM_PATCHSET_SHA256}" \
      local.spark3.b12x.upstream="https://github.com/local-inference-lab/b12x" \
      local.spark3.b12x.commit="${B12X_COMMIT}" \
      local.spark3.b12x.upstream-patches="${B12X_UPSTREAM_PATCHES}" \
      local.spark3.b12x.patchset.sha256="${B12X_PATCHSET_SHA256}" \
      local.spark3.flashinfer.upstream="https://github.com/flashinfer-ai/flashinfer" \
      local.spark3.flashinfer.commit="${FLASHINFER_COMMIT}" \
      local.spark3.cutlass.upstream="https://github.com/NVIDIA/cutlass" \
      local.spark3.cutlass.commit="${CUTLASS_COMMIT}"

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake git liburing-dev liburing2 ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=vllm-source . /opt/spark3/src/vllm/
COPY --from=b12x-source . /opt/spark3/src/b12x/
COPY --from=flashinfer-source . /opt/spark3/src/flashinfer/
COPY --from=cutlass-source . /opt/spark3/src/cutlass/

# Install the exact ARM64 CuTe DSL wheel set captured from the live image. The
# wheelhouse is a named build context and is never retained in an image layer.
RUN --mount=type=bind,from=cutlass-dsl-wheels,source=.,target=/wheels,readonly \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall /wheels/*.whl

# The official image carries an older FlashInfer. Build the source tag recorded
# by the live image; EP transports are irrelevant here and are omitted.
RUN python3 -m pip uninstall -y \
      flashinfer-jit-cache flashinfer-cubin flashinfer-python \
    && BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 BUILD_NVEP=0 \
      FLASHINFER_BUILD_NO_PIP=1 \
      python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        /opt/spark3/src/flashinfer

# Preserve ABI-compatible native extensions from the official ARM64 image,
# overlay the pinned Python source, and install B12X from its patched tree.
RUN set -eux; \
    installed=/usr/local/lib/python3.12/dist-packages/vllm; \
    find "$installed" -maxdepth 1 -type f -name '*.so' \
      -exec cp -a '{}' /opt/spark3/src/vllm/vllm/ ';'; \
    mkdir -p /opt/spark3/src/vllm/vllm/vllm_flash_attn; \
    find "$installed/vllm_flash_attn" -maxdepth 1 -type f -name '*.so' \
      -exec cp -a '{}' /opt/spark3/src/vllm/vllm/vllm_flash_attn/ ';'; \
    for package in deep_gemm flashmla fmha_sm100 tml_fa4 triton_kernels; do \
      if [ -d "$installed/third_party/$package" ]; then \
        mkdir -p "/opt/spark3/src/vllm/vllm/third_party/$package"; \
        cp -a "$installed/third_party/$package/." \
          "/opt/spark3/src/vllm/vllm/third_party/$package/"; \
      fi; \
    done; \
    python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
      --force-reinstall /opt/spark3/src/b12x

ENV PYTHONPATH=/opt/spark3/src/vllm:/opt/spark3/src/b12x \
    TORCH_CUDA_ARCH_LIST=12.1a \
    VLLM_CUTLASS_SRC_DIR=/opt/spark3/src/cutlass \
    TRITON_KERNELS_SRC_DIR=/opt/spark3/src/vllm/vllm/third_party/triton_kernels

FROM source-overlay AS runtime

# R37 changed the caller-owned output schema for this operator. Rebuild only
# the stable extension against the pinned CUDA/Torch/CUTLASS foundation.
RUN cmake -S /opt/spark3/src/vllm -B /tmp/vllm-stable-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
      -DVLLM_STABLE_ONLY=ON \
      -DNVCC_THREADS=2 \
    && cmake --build /tmp/vllm-stable-build \
      --target _C_stable_libtorch --parallel 6 \
    && install -m 0755 /tmp/vllm-stable-build/_C_stable_libtorch.abi3.so \
      /opt/spark3/src/vllm/vllm/_C_stable_libtorch.abi3.so \
    && rm -rf /tmp/vllm-stable-build

RUN set -eux; \
    mkdir -p /tmp/cuda-driver-stub; \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so \
      /tmp/cuda-driver-stub/libcuda.so.1; \
    LD_LIBRARY_PATH="/tmp/cuda-driver-stub:${LD_LIBRARY_PATH}" python3 -c \
      'import importlib.metadata as m; import torch, vllm._C_stable_libtorch; assert m.version("flashinfer-python") == "0.7.0rc1"; assert m.version("nvidia-cutlass-dsl") == "4.6.2"; schema=str(torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.default._schema); print(schema); assert "q_out" in schema and "q_head_padded" not in schema and schema.endswith("-> ()")'; \
    rm /tmp/cuda-driver-stub/libcuda.so.1; \
    rmdir /tmp/cuda-driver-stub

ENTRYPOINT ["/usr/bin/python3", "-m", "vllm.entrypoints.cli.main", "serve"]
