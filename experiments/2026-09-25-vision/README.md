# Vision on the r3 stack

DeepSeek V4.1 Flash ships a 0.90 GiB BF16 vision tower (a 32-layer ViT and
an aligner). The LIL vLLM/B12X stack loads it replicated on every rank, and
the promoted configuration skipped it with `--language-model-only`. This arm
changes only the configuration of the promoted r3 image:

- `--language-model-only` removed, `--limit-mm-per-prompt '{"image":4}'`;
- `--mm-processor-cache-gb 0`: vLLM's default 4 GiB host-memory
  preprocessing cache would come out of dgx1's guard margin;
- KV cache 1.5 GiB per rank (from 2 GiB) to pay for the tower, about 700K
  tokens;
- maximum model length 131,072 (from 160,000), which keeps about 5.3 full
  contexts in that cache and bounds long-prompt transients.

Checks (`run.sh`):

1. The start stays within the 5 GiB startup and 3 GiB steady memory guards.
2. `vision_check.py` describes two images with known contents: shapes with
   text, and a text-heavy benchmark card.
3. The quick bench confirms text quality and decode speed.

Any failure restores the promoted service.

## Results

- **Attempt 1** (KV 1.5 GiB): dgx1 was stopped by the 5 GiB startup guard
  (exit 137, not an OOM kill) about 10 s after engine initialization. Model
  loading took 97.88 GiB against 96.98 GiB text-only, the expected +0.9 GiB
  tower.
- **Attempt 2** (same, with `memtrace.sh`): dgx1 sat at 5.6-6.3 GiB free
  through startup. About 3 s after the API server reported "Application
  startup complete", free memory fell 0.8 GiB in 0.4 s and the guard
  stopped the start. Process RSS was flat (worker 4.2 GB, API server 1.4 GB,
  engine core 1.3 GB), so the drop is a device (unified-memory) allocation.
- **Attempt 3:** KV 1.0 GiB, to measure the full transient and the
  steady-state floor before sizing the cache back up.
