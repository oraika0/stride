# Reference snapshots — directed vs undirected metrics

Two frozen copies of `manager.py` and `simple_monitor.py`, taken either side of
the March 2026 change that made the controller report per-direction link metrics.

```
undirected/   before  — capacity 2C, load tx+rx, one number per physical link
directed/     after   — capacity  C, load tx only, one number per direction
```

They exist so [`docs/directed_vs_undirected_metrics.md`](../../docs/directed_vs_undirected_metrics.md)
has running code to point at. The note argues that undirected aggregation hides
asymmetric congestion; these two files are what that argument is about.

**Nothing here is live and nothing imports it.** The controller runs
`utils/manager.py` and `utils/simple_monitor.py`, which supersede both snapshots:
the current `manager.py` emits the undirected pipeline unchanged *and* the
directed one alongside it, into separate files. Choosing between the two is a
read-side decision — the `_NX_directed` metric suffix — not a matter of swapping
these files in.

Do not copy them back over `utils/`. Doing so would silently drop the directed
outputs that every current figure and metric depends on.
