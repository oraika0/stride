> **Oracle K-prefix curve (32node 144tm test TMs, directed single-path ILP on the canonical frozen k_paths prefix; K=1 = SP closed-form, K=20 = paper_table frozen incumbent, K=2..15 solved at 300 s). All values in percent (offered-load MLU ×100). Reference = per-TM best across all K (at a fixed time limit large-K incumbents can sit above small-K proven optima); LP column = UNRESTRICTED edge-based multicommodity LP (all possible paths, splitting allowed — paper/bounds/edge_lp_bound.py), i.e. the absolute theoretical floor. ✓ = within 0.5% of the per-TM best.**

| | K=1 (SP) | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=10 | K=12 | K=15 | K=20 | LP bound |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| TM06 | 148.2% | 102.8% | 84.9% | 80.4% ✓ | 80.4% ✓ | 80.4% ✓ | 80.4% ✓ | 80.4% ✓ | 80.4% ✓ | 80.4% ✓ | 80.4% ✓ | 80.8% ✓ | 80.4% |
| TM41 | 146.9% | 104.4% | 87.6% | 61.1% ✓ | 61.0% ✓ | 61.0% ✓ | 61.0% ✓ | 61.0% ✓ | 61.0% ✓ | 61.0% ✓ | 61.0% ✓ | 61.5% | 61.0% |
| TM73 | 107.3% | 57.8% | 53.1% | 51.9% ✓ | 51.9% ✓ | 51.9% ✓ | 51.9% ✓ | 51.9% ✓ | 51.9% ✓ | 51.9% ✓ | 51.9% ✓ | 52.7% | 51.9% |
| TM108 | 81.6% | 57.7% | 47.4% | 47.2% | 46.9% ✓ | 46.9% ✓ | 46.9% ✓ | 46.9% ✓ | 46.9% ✓ | 46.9% ✓ | 47.0% ✓ | 47.5% | 46.8% |
| TM141 | 99.6% | 50.5% | 49.1% | 47.3% | 46.3% | 46.0% | 45.9% ✓ | 46.2% | 46.1% | 45.7% ✓ | 46.5% | 46.7% | 45.4% |
| **平均最優 MLU** | 116.7% | 74.6% | 64.4% | 57.6% | 57.3% | 57.2% | 57.2% | 57.3% | 57.3% | 57.2% | 57.4% | 57.8% | 57.1% |
| **達最佳解比例** | 49.5% | 79.6% | 90.8% | **99.2%** | **99.7%** | 99.9% | 99.9% | 99.8% | 99.8% | 100.0% | 99.6% | 98.8% | — |

**Smallest K within 0.5% of the best optimum, per TM**: TM06: K=4, TM41: K=4, TM73: K=4, TM108: K=5, TM141: K=7 → overall K* = 7

> TM06/41/73 的 best 恰等於 LP 下界（證明最優）。TM141 為最難實例，300 s 內未證明收斂，K≥5 的 incumbent 在 45.7%–46.5% 間抖動（皆為求解器噪聲，非真實的 K 效應），其 K*=7 由嚴格 0.5% 規則判定；實質上 K=4 已達整體最優的 99.2%、K=5 達 99.7%。K=20 列（98.8%）同為 incumbent 噪聲，非真實退化。
>
> **關鍵界定**：LP 欄為「不限候選集、所有可能路徑、允許任意分流」的全圖理論下界（edge-based MCF LP）。20 條候選內的最佳單路徑配置與此極限的差距為 0.00%–0.76%（平均 0.2%）——候選集限制、單路徑限制、以及 K>20 的擴充空間三者合計不足 1%。
