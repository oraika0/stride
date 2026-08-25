> **GPU inference latency of one full routing decision (32-node, M = 8, 992 pairs, K = 20), seeds 17+18**

Timed call: `StrideAgent.get_action()` (encoder + full 8-step chain rollout). Device: NVIDIA GeForce RTX 3060 Ti. 20 warmup calls discarded, 200 timed calls per seed, CUDA-synchronised. Input is a real archived link-state snapshot (`06_15_net_info_directed.csv`, MLU = 73.2%); decoding is greedy with a fixed step count, so latency is shape-determined and not input-dependent.

| seed | checkpoint | median (ms) | mean (ms) | p95 (ms) |
| --: | :-- | --: | --: | --: |
| 17 | `results/stride/runs/base_32node_s17_20260605_114040/train/model` | 16.26 | 16.27 | 16.38 |
| 18 | `results/stride/runs/base_32node_s18_20260605_221156/train/model` | 16.23 | 16.25 | 16.37 |
| **mean** | | **16.25** | | |

**Measured latency: ~16.2 ms**

Regenerate with `python make_inference_bench.py`.
