# Testbed difficulty — offered-load SP / ILP (train+test)

| Testbed | N | SP (OSPF) MLU | ILP MLU | SP–ILP gap (%) | SP MLU > 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GÉANT (s3) | 20 | 0.87 ± 0.15 | 0.67 ± 0.10 | 30 ± 10 | 4/20 (20%) |
| 32-node (144tm) | 91 | 1.25 ± 0.23 | 0.61 ± 0.10 | 106 ± 21 | 81/91 (89%) |

> Offered-load (theoretical) SP / ILP MLU over the experiment TMs (train+test only); can exceed 1.0 and is **not** comparable to the real-Mininet measured MLU. ILP is single-path at K=20 (the agent's candidate set), directed, 300 s time limit. mean ± sample std (ddof=1). Per-TM LP (flow-split lower bound) is in the *_lp_ilp.csv files.
