# Decision

- **Asynchronous disk Engram rows:** accept for productization, pending the
  owner's promotion decision. Before promotion:
  - carry the overlay as a vLLM patch in `patches/vllm/series`;
  - build the image with `bin/spark3 build`;
  - confirm the gain with alternating boots of each arm (single boots vary
    by about 3%).
- **Draft-only FP8 LM head:** reject. It saves about 0.8 ms per step but
  loses about 2.5% of accepted drafts; the net is neutral.
- **DSpark depth 5:** reject at adaptive-verification cost scales 2.0 and 4.0.
  The fixed cost of drafting five positions outweighs the acceptance gain on
  prose.
- **Follow-up:** make adaptive verification's startup cost profile stable
  across boots (persist it, or profile longer). It currently moves throughput
  by about 3% from boot to boot.
