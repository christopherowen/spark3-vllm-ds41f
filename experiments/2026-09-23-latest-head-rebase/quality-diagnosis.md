# Incremental generation quality gate

2026-09-23. The consolidated managed-weight image serves on all three ranks,
but a temperature-zero LRU request can produce malformed Python identifiers.
The full request and response receipts are in `runs/consolidated-managed/`.

The exact user prompt is: “Write a short Python LRU cache class with get and
put methods using OrderedDict. Give code and one sentence on complexity.”
It is 29 prompt tokens with `chat_template_kwargs={"thinking":false}`. On the
same running image, independent requests generated `valueerk`, `valueazon`,
`value recon`, and `value splash` after a previously valid `def put(self, key,
value` prefix; another repeat generated valid code. The same defect occurs
through `/v1/completions` when given the 29 chat prompt token IDs. It is
intermittent even at temperature zero and seed 42, and is not specific to chat
response formatting.

The server returned the prompt and generated token IDs for a failing request.
Passing the **identical 114-token sequence** (29 prompt plus 85 generated token
IDs, ending in ` value`) to `/v1/completions` as one prompt chose `):\n` with
logprob -0.00103. The original stepwise decode chose ` recon`; its top five
tokens were all stray words or suffixes. A second failing request chose
` splash`, while a forced continuation from its exact token-ID prefix produced
15 valid subsequent tokens beginning `):\n        if key in self.cache:`. An
earlier five-point comparison from a successful request matched stepwise and
forced next tokens at generated prefix lengths 1, 16, 32, 64, and 84. This
localizes the fault to state accumulated during some incremental requests,
without yet identifying which component carries the bad state.

The B12X attention adapter, its cache writers, vLLM's decode metadata and
Engram lookback/disk preparation all cross this boundary. Source inspection
confirms that the compressed-cache block table fix remains in the consolidated
adapter. It does not establish that the runtime tables or writes are correct.
The legacy B12X compressed-MLA test helper was written for an older plan API.
The current prepared-API test passed a synthetic V4.1 packed-cache FP8 decode
comparison with 24 local heads under a 4 GiB cap on dgx3. It uses 64-token
pages, not this service's production 128-token SWA pages, and cannot validate
the live metadata, writers, disk Engram, or full model. A synthetic V2 Engram
hash comparison also matched whole-sequence and incremental chunks. The
separate model-state test fixture predates this branch's disk-Engram extension
and lacks `disk_engram_models`; it failed before exercising the gather. These
bounded results and inconclusive setup failures are recorded in
`runs/capped-component-controls.json`.
The current B12X TP3 disk-Engram lookup test also passed varied rows, graph
replay, and stream reuse with production-style nonresident scales. This checks
the isolated lookup, not the live input hashes or synchronization across ranks.

No image or service restart was needed for these checks. The candidate remains
unpromoted and guarded. Before changing throughput settings, the next causal
experiment must compare the same prompt and token-ID boundary against one
alternate generation-state path, with a fixed image/configuration and a
memory-guarded launch if a runtime change is required. Preserve both good and
bad request receipts, and do not treat one valid repeat as a quality pass.
