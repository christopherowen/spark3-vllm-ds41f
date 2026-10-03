# Current state

Promoted 2026-10-02 as
[`2026-10-02-karmic-kraken-r5o-64k`](../manifests/baselines/2026-10-02-karmic-kraken-r5o-64k.json)
and running on all three nodes from `config/cluster.json`.

| Setting | Active value |
|---|---:|
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `04c30fa9` + patches 0001-0026 (0005-0009, 0011 and 0026 off by default; 0023 off in configuration), B12X `f8069b2c` + switchless RoCEnante, CuTe DSL 4.7.1 pin, top-k position-tie, dense GEMM and prefill stage-fence patches, NCCL 2.30.7 + IB send-path fence |
| Image | `vllm-ds41f-kkref:04c30fa98e79-r5o`, one digest on all ranks, built by `bin/spark3 build` |
| Hosts | DGX Spark 26.09.2, kernel `7.0.0-1019-nvidia-64k` with `kho=off`, driver 580.178.04, no desktop |
| Tensor parallel ranks | 3 |
| Maximum model length | 524,288 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |
| Indexer under sequence parallelism | each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025) |
| Indexer top-k ties | lowest logical position (B12X 0003): selections repeat exactly |
| Dense GEMM stage release | async-proxy fence before the TMA refill (B12X 0004): the shared expert no longer returns wrong columns beside the routed MoE |
| Prefill stage releases | async-proxy fence before the TMA refill in the BF16 and mHC prefill projections and contiguous attention (B12X 0005) |
| Explicit KV memory | 3.5 GiB per rank |
| Reported KV capacity | 2,845,543 tokens (5.43x full 512K windows) |
| Memory-saver | signed DKMS 0.2.0 on all nodes; loaded UVM leaf-table packing enabled |
| Page-size profiles | 64 KiB selected; 4 KiB retains 2.2 GiB KV and a 262,144-token limit |
| Display carve-out | embedding and output-head weights (842.5 MiB per rank) in the firmware scanout reserve (`SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, the GPU's DRM card by PCI path, `/dev/dri/by-path/pci-000f:01:00.0-card`, as `/dev/dri/card0`); the DRM file closes after the import, so the text console keeps drawing |
| Kernel backend | B12X (the default; neither `kernel_backend` nor `VLLM_DS41_KERNEL_BACKEND` is set); TileLang is a [candidate](kernel-backends.md) |
| Sparse-attention arithmetic | B12X's tuned choice (`VLLM_DS41_ATTENTION_COMPUTE=auto`); BF16 (patch 0023) is available and off pending a fidelity test |
| Image input | vision tower loaded, up to 4 images per request, no host preprocessing cache |
| DSpark | 5 draft tokens, draft TP 3, adaptive verification (cost scale 2.0), dead verification rows below survival 0.2, block rejection; vocabulary-parallel greedy drafts, NVFP4 drafter head and Markov projection |
| CUDA graphs | full, capture sizes 1-48 |
| B12X W4A8 tiny decode | disabled (`B12X_W4A8_TINY_DECODE=0`) |
| Engram projection | sharded across ranks (`projection_tp`) |
| Engram rows | read beside the forward launch (`SPARK3_ENGRAM_ASYNC=1`); base overlap off (`VLLM_DS41_ENGRAM_OVERLAP=0`) |
| L2 weight prefetch | on (base default on SM121; runs since r5i) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s); hosts set `vm.watermark_boost_factor=0` |
| Async scheduling | enabled |
| Reasoning | enabled by default; a request's `thinking` or `enable_thinking` is honored |

Serving validation and capacity evidence are in the
[memory-saver deployment](../experiments/2026-10-02-memory-saver-capacity/README.md).
The image, weights and model arithmetic are unchanged from r5o.

The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [replicate.md](replicate.md).
