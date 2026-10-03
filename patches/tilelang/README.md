# TileLang patch stack

Base: `tile-ai/tilelang@b95ee4ff` (main, "Support bfloat16 scalar parameters
(#3361)"), TileLang 0.1.15. The image builds it from source for SM121 and
replaces the base image's `tilelang` wheel. It is a build input only when
the cluster profile selects the TileLang kernel backend
(`docs/kernel-backends.md`). The DS4.1 kernels themselves are part of the vLLM
patch stack; these patches add the SM120-family instructions they need.

- `0001-sm121-block-scaled-mma.patch` enables the SM120 block-scaled MMA
  templates on SM121. The NVFP4 guard tested only
  `CUTLASS_ARCH_MMA_SM120A_ENABLED`, which `sm_121a` does not define, so the
  instruction compiled to a trap on GB10. The SM12x tests now run on any
  SM12x device.
- `0002-sm120-mxf8f6f4-block-scaled-mma.patch` adds
  `mma.sync.kind::mxf8f6f4.block_scale` (MXFP8 and mixed E4M3/E2M1 operands,
  UE8M0 scales per 32) to `T.mma_gemm_blockscaled`, and the
  `ldmatrix .b8x16.b4x16_p64` variant that expands packed E2M1 held in
  `float4_e2m1_unpacked` shared memory. These carry the block-FP8 linears and
  the FP8 x FP4 routed experts.
- `0003-sm120-mxfp4-block-scaled-mma.patch` adds
  `kind::mxf4nvf4.block_scale.scale_vec::2X` with UE8M0 scales (MXFP4), which
  the indexer's FP4 scores use.

Applying them to the base yields patch head `30d854b6` and tree `1697ad52`.
The submodules (`3rdparty/tvm` and `3rdparty/cutlass`, with TVM's own) are
unchanged; the source manifest records their commits.

The patches are the first three commits of branch `deepseek-v41-sm120` in the
contribution fork `christopherowen/tilelang` (tree `1697ad52` at its third
commit, `193a2c17`). The branch's later commits add the same kernels as
TileLang examples and are not build inputs. None is upstream yet.
