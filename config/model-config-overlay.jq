# Adapt the upstream DeepSeek V4.1 Flash model config to TP3 without changing
# weights. The padded heads/groups are virtual; vLLM uses virtual_heads_from to
# preserve the original 64-head / 8-group geometry.
.text_config.num_attention_heads = 72
| .text_config.o_groups = 9
| .text_config.virtual_heads_from = {
    "num_attention_heads": 64,
    "o_groups": 8
  }
| .virtual_heads_from = {
    "num_attention_heads": 64,
    "o_groups": 8
  }
| .native_tp3_overlay = {
    "description": "64 real heads / 8 output groups padded to 72 / 9 for TP3",
    "base_weights": "unchanged native MXFP4/MXFP8"
  }
