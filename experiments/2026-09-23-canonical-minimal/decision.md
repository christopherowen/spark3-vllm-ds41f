# Decision

Status: **retain for more evidence**

The weekend setup is the speed target but not a quality reference. It fails
the fixed LRU gate 4/5 with the same malformed tokens as the canonical-main
consolidated candidate. The canonical-main rebase did not introduce the
defect. The defect is in a component the two stacks share and appears only
in incremental decode.

Ruled out: RoCEnante all-reduce (NCCL also fails), and B12X's FP8-internal /
8-head-block split decode (BF16/H16 also fails). Not yet runnable as a
control: canonical FlashInfer attention on this image, at either block size.

Remaining shared candidates: B12X DS4.1 cache writers and decode metadata
consumption, the lightning-indexer decode top-k, disk Engram row staging,
B12X MoE/MXFP8, and the TP3 virtual-head geometry. The next step is a guarded
per-layer comparison of stepwise decode against one-shot prefill for the same
position, to find the first diverging layer and component.

Do not promote. Speed recovery on the canonical-main candidate (CUDA graphs,
then DSpark) is independent of this defect. It can proceed in parallel, with
the LRU gate recorded honestly for both arms.
