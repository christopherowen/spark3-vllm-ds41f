# Vision on the r3 stack

DeepSeek V4.1 Flash ships a 0.90 GiB BF16 vision tower (a 32-layer ViT and
an aligner). The LIL vLLM/B12X stack loads it replicated on every rank, and
the promoted configuration skipped it with `--language-model-only`. This arm
changes only the configuration of the promoted r3 image:

- `--language-model-only` removed, `--limit-mm-per-prompt '{"image":4}'`;
- `--mm-processor-cache-gb 0`: vLLM's default 4 GiB host-memory
  preprocessing cache would come out of dgx1's guard margin;
- KV cache below the promoted 2 GiB per rank to pay for the tower (sized per
  attempt below);
- maximum model length 131,072 (from 160,000), which bounds long-prompt
  transients.

Checks (`run.sh`):

1. The start stays within the 5 GiB startup and 3 GiB steady memory guards.
2. `vision_check.py` describes two images with known contents: shapes with
   text, and a text-heavy benchmark card.
3. The quick bench confirms text quality and decode speed (`run.sh --idle
   SECONDS` traces the idle service instead).

A failure leaves the cluster stopped; nothing rolls back automatically.
`memtrace.sh` records dgx1 memory from the start until the checks finish.

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
- **Attempt 3** (KV 1.0 GiB, 410,919 tokens, 3.1 full contexts): started.
  dgx1 free memory fell to 5.04 GiB at the end of startup: 0.8 GiB during
  graph capture and 0.3 GiB more as the API server came up. No process RSS
  grew and no other process appeared, so this memory is outside every
  process's RSS. The next idle reading, 14 minutes later, was 6.0 GiB, and it
  stayed at 6.0 under 8-way decode and a 23K-token prefill. The worker's CUDA
  allocation was 104,745-104,753 MiB throughout.
  - Both image checks passed. Shapes: "a solid red circle on the left and a
    solid blue square on the right", text "SPARK3 42". Card: DeepSeek V4.1
    Flash, 173.
  - The quick bench matched r3 at every point (all within the 95%
    intervals; single-stream steps 50.3/52.1 ms vs 50.7/51.2 ms) and passed
    quality 5/5. Minimum free memory: dgx1 5.01, dgx2 6.44, dgx3 7.19 GiB
    (r3: 6.28, 7.11, 7.84).
- **Attempt 4** (same configuration, `run.sh --idle 600`, every meminfo
  field and the worker's CUDA allocation traced): started; lowest dgx1
  reading 5.29 GiB, during `doctor` right after the switch to the steady
  guard. The worker's CUDA allocation settled at 104,695-104,733 MiB. That
  is r3's with KV 2.0 to within the measurement: the tower's 0.9 GiB cost
  what the smaller cache saved.
  - Most of the startup dip is not memory use. MemAvailable swung between
    two levels while free memory, file pages and reclaimable slab stayed
    put: at 10:06:04-06 it fell 1,649 MiB while free fell 774 MiB and no
    other field moved more than 30 MiB. MemAvailable minus (free + file LRU
    + KReclaimable) alternated between -1,114 MiB and about -1,990 MiB.
  - The cause is the kernel's watermark boost. `vm.watermark_boost_factor`
    is the default 15000, so after a fragmentation event (the nodes show
    2.8M compaction stalls) the Normal zone's low watermark rises by up to
    1.5 x its 291 MiB high watermark, 437 MiB. MemAvailable subtracts the low
    watermark once from page cache and once from reclaimable slab, so a full
    boost hides 875 MiB, matching the 876 MiB swing. kswapd then reclaims and
    the boost clears. Under steady load (8-way decode, a 23K-token prefill,
    both image checks) the residual held at -1,114 MiB with no boost.
  - Excluding the boost, dgx1's lowest reading was about 6.1 GiB, so the
    real margin above the 5 GiB guard is about 1.1 GiB with KV 1.0.
