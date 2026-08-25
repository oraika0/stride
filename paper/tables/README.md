# paper_table — offered-load SP / LP / ILP theoretical-MLU tables

Self-contained source for Appendix A / §5.7.2 (testbed difficulty). One builder,
one frozen data folder, clear provenance.

## Layout

```
data/
  ilp_geant_k20.json     ILP single-path min-MLU, K=20, directed   (GÉANT s3, 20 TMs)
  ilp_32node_k20.json    ILP single-path min-MLU, K=20, directed   (32-node 144tm, 144 TMs)
  ilp_32node_k20.log     full per-TM solver log for the 32-node run
build_paper_table.py     builder — no arguments
geant_lp_ilp.csv         OUTPUT  per-TM  tm,split,total_demand_mbps,sp_mlu,lp_mlu,ilp_mlu
32node_lp_ilp.csv        OUTPUT  per-TM  (all 144 rows)
lp_ilp_summary.csv       OUTPUT  SP + ILP stats per topo (train+test subset)
lp_ilp_summary.md        OUTPUT  human-readable
```

## What is frozen vs recomputed

- **ILP (frozen in `data/`)** — single-path min-MLU at **K=20** (the same candidate
  set the DRL agent picks from), directed, **300 s** time limit per TM. On 32-node
  this is a *time-limited incumbent*, not proven optimal, so re-solving can shift a
  TM by a few thousandths. It is therefore stored as data, not recomputed.
- **SP and LP (recomputed by the builder every run)** — SP = candidate index 0 for
  every pair; LP = flow-split lower bound at K=20. Both are K-independent / cheap and
  fully deterministic, so they need no frozen data. The builder imports the solver
  from `paper/bounds/check_min_mlu.py` (not duplicated here).
- **Scope** — the summary aggregates the experiment's train+test TMs (from
  `config/env/{geant,32node_144tm}_config.py`): GÉANT 20 (15+5), 32-node 91 (86+5).
  The 32-node per-TM CSV keeps all 144 rows; the summary uses the 91 subset.

## Rebuild the tables (fast)

```bash
cd ~/stride
conda activate stride
python paper/tables/build_paper_table.py
```

## Regenerate the ILP data itself (slow — hours on 32-node)

Only needed if the dataset / candidate paths change. From the repo root:

```bash
# GÉANT (scale 3), 20 train+test TMs
python paper/bounds/check_min_mlu.py --topo geant --scale 3 --no-split \
    --max-k 20 --time-limit 300 --save-json paper/tables/data/ilp_geant_k20.json

# 32-node (144tm), all 144 TMs, 6 workers
python paper/bounds/check_min_mlu.py --topo 32node --variant 144tm --no-split \
    --max-k 20 --time-limit 300 --workers 6 --all-tms \
    --save-json paper/tables/data/ilp_32node_k20.json
```

LP is also K=20 single-set; it is NOT shown in the paper (SP + ILP only) but is kept
in the per-TM CSVs for completeness, since SP–LP and SP–ILP gaps are near-identical
(LP ≈ ILP on both testbeds).
