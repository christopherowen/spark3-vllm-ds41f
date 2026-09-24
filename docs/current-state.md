# Current state

Promoted 2026-09-24 as
[`2026-09-24-karmic-kraken-nofiat`](../manifests/baselines/2026-09-24-karmic-kraken-nofiat.json)
and running on all three nodes from `config/cluster.json`.

| Setting | Active value |
|---|---:|
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `01f1b874` and B12X `0f846212` + switchless RoCEnante patch |
| Image | `vllm-ds41f-kkref:01f1b874c774-r1`, one digest on all ranks |
| Tensor parallel ranks | 3 |
| Maximum model length | 160,000 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Explicit KV memory | 2 GiB per rank |
| Reported KV capacity | 933,168 tokens (5.83x full 160K windows) |
| DSpark | 3 draft tokens, draft TP 3, adaptive verification (cost scale 2.0) |
| CUDA graphs | full, capture sizes 1-32 |
| B12X W4A8 tiny decode | disabled (`B12X_W4A8_TINY_DECODE=0`) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s) |
| Async scheduling | enabled |
| Reasoning | enabled by default |

Measured on this configuration: LRU coherence gate 5/5; code at eight streams
172 tok/s and single-stream prose/code about 41/52 tok/s; cold prefill about
4,100 tok/s; dgx1 minimum MemAvailable about 7 GiB. The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [replicate.md](replicate.md).
