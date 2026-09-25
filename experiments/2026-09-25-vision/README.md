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
  - A boost does not always clear quickly. At 10:11:58, with the service
    idle, the residual moved to -1,989 MiB and stayed. /proc/zoneinfo then
    showed the Normal zone boost at 111,946 pages (437 MiB, the maximum) with
    MemAvailable at 5.08 GiB. Either kswapd did not run to reset it or new
    fragmentation events kept raising it again; the trace cannot tell which.
  - Excluding the boost, dgx1's lowest reading was about 6.1 GiB, so the
    real margin above the 5 GiB guard is about 1.1 GiB with KV 1.0. With
    the boost, a start can land anywhere from that margin down to about
    0.2 GiB, which is why attempts 1 and 2 were stopped.
  - One real transient came from outside the stack: dgx1's user-level
    `law-desire-reconciler` (a 20-second timer) coincided with a 1 GiB dip
    lasting 0.4 s at 10:07:58.

- **Attempt 5** (KV 1.4 GiB, 575,304 tokens, 4.39 contexts of 131,072), after
  the owner set `vm.watermark_boost_factor=0` on all three nodes (live and in
  `/etc/sysctl.d/90-watermark-boost.conf`); the guards stayed at 5 and 3 GiB.
  - The Normal zone boost read 0 in all 160 trace samples. dgx1's lowest
    reading during startup was 6.21 GiB, 1.2 GiB above the startup guard
    (predicted about 5.7).
  - Both image checks passed (531 and 1,017 prompt tokens, 1.6 and 1.3 s).
  - Quick bench against r3: equal at every decode point, quality 5/5;
    single-stream steps 49.7/51.5 ms vs 50.7/51.2 ms. Lowest free memory
    dgx1 5.65, dgx2 6.66, dgx3 6.80 GiB.
  - Capacity suites against r3: prefill 2K 3,882 (+1.4%), 32K 4,174, 64K
    4,072 tok/s (both within noise; the 128K point no longer fits the
    131,072-token limit); 32K prefix replay 7.62 s cold, 0.26 s warm; four
    64K contexts admitted, peak KV 31%, 12.3 tok/s per stream. Lowest free
    memory dgx1 5.64 GiB. dgx3's GPU peaked at 79 C during prefill (10 C
    below its limit), as in earlier runs; no thermal slowdown.

## Decision

Promoted as `2026-09-25-karmic-kraken-r3-vision`: the configuration above
under the promoted container name, on the unchanged r3 image. Text
performance is unchanged, images work, and dgx1 keeps a 1.2 GiB startup
margin. Hosts now need `vm.watermark_boost_factor=0` (docs/replicate.md).
