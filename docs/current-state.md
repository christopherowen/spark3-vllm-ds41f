# Current state

Promoted 2026-09-25 as
[`2026-09-25-karmic-kraken-r3-vision`](../manifests/baselines/2026-09-25-karmic-kraken-r3-vision.json)
and running on all three nodes from `config/cluster.json`.

| Setting | Active value |
|---|---:|
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `01f1b874` + Engram projection TP and asynchronous Engram row patches, B12X `0f846212` + switchless RoCEnante patch |
| Image | `vllm-ds41f-kkref:01f1b874c774-r3`, one digest on all ranks, built by `bin/spark3 build` |
| Tensor parallel ranks | 3 |
| Maximum model length | 131,072 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Explicit KV memory | 1.4 GiB per rank |
| Reported KV capacity | 575,304 tokens (4.39x full 131K windows) |
| Image input | vision tower loaded, up to 4 images per request, no host preprocessing cache |
| DSpark | 3 draft tokens, draft TP 3, adaptive verification (cost scale 2.0), block rejection |
| CUDA graphs | full, capture sizes 1-32 |
| B12X W4A8 tiny decode | disabled (`B12X_W4A8_TINY_DECODE=0`) |
| Engram projection | sharded across ranks (`projection_tp`) |
| Engram rows | read beside the forward launch (`SPARK3_ENGRAM_ASYNC=1`) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s); hosts set `vm.watermark_boost_factor=0` |
| Async scheduling | enabled |
| Reasoning | enabled by default |

Measured on this configuration: LRU coherence gate 5/5; both image checks
pass; code at eight streams 176 tok/s and single-stream prose/code about 41/53
tok/s; cold prefill about 4,100 tok/s; dgx1 minimum MemAvailable 6.2 GiB during
startup and 5.6 GiB under load. The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [replicate.md](replicate.md).
