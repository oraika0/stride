> **Per-step training time (`step_time_sec` and `training_time_sec`, seeds 17+18)**

`step_time` = whole loop iteration (env + inference + update), every step. `update` = `agents.update()` only, averaged over steps where it fired (the first 66 warmup steps log 0.0 and are excluded). `env+inference` = `step_time - update`. Seed mean = unweighted mean of the two per-run means.

| config | seed | run | host | steps | warmup | step_time (s) | update (s) | env+inference (s) |
| :-- | --: | :-- | :-- | --: | --: | --: | --: | --: |
| GÉANT M=8 | 17 | `aeokwfnv` | pc2 | 3000 | 66 | 2.783 | 2.684 | 0.158 |
| GÉANT M=8 | 18 | `gx5d5942` | pc0 | 3000 | 66 | 3.182 | 3.095 | 0.155 |
| **GÉANT M=8** | **mean** | | | | | **2.98** | **2.89** | **0.156** |
| 32node M=4 | 17 | `qmb52n8x` | pc1 | 3000 | 66 | 2.753 | 2.591 | 0.219 |
| 32node M=4 | 18 | `8ueo308d` | pc1 | 3000 | 66 | 2.739 | 2.579 | 0.218 |
| **32node M=4** | **mean** | | | | | **2.75** | **2.58** | **0.218** |
| 32node M=6 | 17 | `apmfzim9` | pc2 | 2957 | 66 | 3.412 | 3.272 | 0.213 |
| 32node M=6 | 18 | `xq3wk44p` | pc0 | 3000 | 66 | 3.966 | 3.827 | 0.223 |
| **32node M=6** | **mean** | | | | | **3.69** | **3.55** | **0.218** |
| 32node M=8 | 17 | `jkutslaf` | pc1 | 3000 | 66 | 4.261 | 4.129 | 0.223 |
| 32node M=8 | 18 | `0dwf7m68` | pc2 | 3000 | 66 | 4.170 | 4.040 | 0.220 |
| **32node M=8** | **mean** | | | | | **4.22** | **4.08** | **0.221** |
| 32node M=10 | 17 | `si9kqpfn` | pc1 | 2998 | 66 | 5.048 | 4.925 | 0.231 |
| 32node M=10 | 18 | `u0vr1vuv` | pc1 | 3000 | 66 | 5.045 | 4.923 | 0.231 |
| **32node M=10** | **mean** | | | | | **5.05** | **4.92** | **0.231** |
| 32node M=12 | 17 | `y0ew5pqf` | pc2 | 3000 | 66 | 5.700 | 5.595 | 0.228 |
| 32node M=12 | 18 | `pvz1yyi7` | pc1 | 3000 | 66 | 5.793 | 5.688 | 0.230 |
| **32node M=12** | **mean** | | | | | **5.75** | **5.64** | **0.229** |
