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

Pending.
