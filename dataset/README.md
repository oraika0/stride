# Dataset

Everything the agent and the network read at run time: the topologies, the
traffic matrices, the frozen candidate-path sets, and the scripts that build
them. Nothing here is written during training.

## Layout

```text
<topo>_traffic/
├── bw_r.txt                    the topology: one directed link per line
├── k_paths.json                the frozen K=20 candidate paths per OD pair
├── traffic_generator/          raw traffic matrices + the generator that made them
└── <topo>_<n>tm/  or  23node_s<scale>/
    └── TM-<id>/{Clients,Servers}/*.sh    the iperf3 scripts Mininet replays
```

`bw_r.txt` is `node1, node2, <unused>, capacity_mbps`; the third column is
ignored. Mininet builds the topology from it, and the metric pipeline uses it as
the capacity reference.

## What is in the repository, and what is not

Roughly 8 MB is tracked — what cannot be rebuilt, plus the sources for what can:

| tracked | why |
| --- | --- |
| `bw_r.txt` | the topology itself |
| `k_paths.json` | the frozen candidate set; **not regenerable**, see below |
| `k_paths_k30_ext.json` | the K=30 extension, whose first 20 entries must stay verbatim |
| `traffic_generator/traffic-matrices/` (96 XML), `*.pkl` | what `prepare_dataset.py` reads |
| `ilp_actions_*.json`, `*_mlu_by_tm.json` | small reference values |

### Where the GÉANT traffic matrices come from

The public archive is four months of measurements at 15-minute intervals. The
paper uses one day of it, hourly, and the narrowing happens in two steps:

```
10,772 XML   2005-01-01 .. 2005-04-29, every 15 min   traffic-matrices-all/  (207 MB)
     96 XML   2005-01-03 only, every 15 min           traffic-matrices/      (1.9 MB, tracked)
     24 TM    2005-01-03 only, hourly                 23node_s3/             (generated)
```

The first step is a choice about which day; the second is
`utils/iperf3_geant.py:_collect_perhour_xmls`, which keeps only the files ending
`-00.xml`. Twenty-four is what `config/env/geant_config.py` declares as `num_tm`.

The 96 quarter-hourly files are tracked rather than just the 24 the paper uses,
because they cost 1.4 MB more and keep the option of a denser sampling open. The
full archive is not tracked: it is the public GÉANT/TOTEM dataset, so it can be
fetched rather than redistributed, and it is larger than everything else here put
together.

Ignored, because `prepare_dataset.py` rebuilds them or nothing uses them:
the per-TM iperf3 script trees (`23node*/`, `32node_*tm/`, ~54 MB),
`traffic-matrices-all/` (the complete GÉANT archive, 207 MB — see above), and the
notebooks. `NEW_Geant_s3_perhour/` was 507 MB that no config referenced; it was
deleted from disk on 2026-08-19, and `utils/iperf3_geant.py` still rebuilds it.

So a fresh clone runs step 7 of [`../QUICKSTART.md`](../QUICKSTART.md) and has
everything it needs.

## Scripts

Run them from the repository root.

| script | what it does |
| --- | --- |
| [`prepare_dataset.py`](prepare_dataset.py) | traffic matrices → the per-host iperf3 scripts the real environment replays |
| [`get_k_paths.py`](get_k_paths.py) | topology → `k_paths.json`, the agent's action space |
| [`extend_k_paths.py`](extend_k_paths.py) | K=30 candidate file that keeps the frozen K=20 prefix verbatim — read its docstring before touching candidate sets |
| [`generate_scaled_tms.py`](generate_scaled_tms.py) | scaled copies of the GÉANT demands (the paper uses scale 3) |
| [`compute_ilp_actions.py`](compute_ilp_actions.py) | ILP-optimal path index per OD pair, the oracle baseline |
| [`convert_32node_to_enero.py`](convert_32node_to_enero.py), [`convert_geant_tm_to_enero.py`](convert_geant_tm_to_enero.py) | traffic matrices → the `.demands` format the vendored gym environment reads |

```bash
python dataset/prepare_dataset.py --topology 32node --tms 144tm
python dataset/prepare_dataset.py --topology 32node --tms 24tm
python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3
```

## `k_paths.json` is frozen

It holds exactly 20 paths per pair in a fixed order, and it is the DRSIR
authors' original artifact — byte-identical to their release. It was generated
with a hop-count Yen recipe whose tie-breaking depended on their live
controller's adjacency order, and that ordering is lost, so **the file cannot be
regenerated**: a local replay reproduces 107 of 992 pairs on 32-node.

Every reported result indexes into this ordering, and the K ablation is only a
controlled comparison because K=10 and K=15 are prefixes of the same list. If a
larger K is needed, extend the file — `extend_k_paths.py` appends positions
20–29 and leaves 0–19 untouched — and never regenerate it.

`get_k_paths.py` is what would build the file for a *new* topology, not a
replacement for the existing one.

## Adding a topology

1. Create `<name>_traffic/` here.
2. Put the topology in `<name>_traffic/bw_r.txt` in the format above.
3. Create the traffic-matrix directory as `<name>_<n>tm/`.
4. Build the candidate paths:

   ```bash
   python dataset/get_k_paths.py --topology <name>
   ```

   It refuses to overwrite an existing `k_paths.json`.

5. Add `config/env/<name>_config.py` pointing at the files. The filename minus
   `_config` is what `--env` takes.
