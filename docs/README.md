# Methodology notes

Why the measurement pipeline, the reward and the problem formulation are built
the way they are. Each note is a self-contained write-up of one investigation;
the code column is what it justifies or corrects.

Originally written in HackMD. The Markdown is the whole record; the PDF exports
that used to sit beside each note were dropped in the 2026-08 cleanup.

| Note | What it settles | Code it applies to |
| --- | --- | --- |
| [`delay_measurement_issues.md`](delay_measurement_issues.md) | LLDP round-trip timing systematically underestimates real queuing delay (steady state ≈ ⅓ of truth); tc backlog is used as ground truth instead | `utils/simple_delay.py`, the Ryu `switches.py` patch (README §2.3), `diagnostics/udp_probe_echo.py` |
| [`loss_measurement_issues.md`](loss_measurement_issues.md) | Why per-link loss read ≈ 0 for months — the tc qdisc layering — and why the fix is netem-only | `utils/simple_tc_loss.py`, `diagnostics/test_queue_fill.py` |
| [`directed_vs_undirected_metrics.md`](directed_vs_undirected_metrics.md) | Undirected link metrics aggregate both directions and hide asymmetric congestion; the raw OpenFlow counters were directed all along | `utils/manager.py`, `utils/simple_monitor.py`, `utils/reference/` |
| [`sim_fluid_queue_calibration.md`](sim_fluid_queue_calibration.md) | The three knobs that align the fluid-queue simulator's delay/loss with real Mininet | `A-Traffic-.../environment16.py`, `test_sim_only.py` |
| [`lp_ilp_analysis.md`](lp_ilp_analysis.md) | LP / ILP / shortest-path optimal MLU per TM — the reference band a trained policy is judged against | `paper/bounds/`, `paper/tables/build_paper_table.py` |
| [`te_objective_design.md`](te_objective_design.md) | Average link utilisation is not a valid optimisation target; MLU is | `algs/stride.py::_compute_reward` |
| [`mdp_formulation.md`](mdp_formulation.md) | Why the problem is a POMDP over link observations rather than an MDP over the traffic matrix, and what follows for the architecture | `algs/stride.py` |
| [`control_delay_is_approximation.md`](control_delay_is_approximation.md) | The importance-sampling ratio compares states one control period apart; why that approximation is acceptable | `algs/stride.py`, `loader/train_loader.py` |
| [`drl_or_s_proactive_vs_reactive.md`](drl_or_s_proactive_vs_reactive.md) | DRL-OR-S installs flow rules proactively, so its per-flow claim is not the reactive first-packet path — why it is not a like-for-like baseline | related work; no code |
| [`negative_result_lu_est_weighting.md`](negative_result_lu_est_weighting.md) | Reconstructing link utilisation from delay growth and loss, and weighting the per-pair actor loss with it — what it was, and the three weightings that all lost to the unweighted baseline | removed 2026-08-19; no code |
| [`negative_result_transferable_heads.md`](negative_result_transferable_heads.md) | Cross-topology transfer was the reason for the architecture; the head is the one component with the pair count in its parameter shape. Eleven replacements that carry no per-pair weights, and the collapse upstream of all of them | `algs/stride.py` `head_type` |
| [`controller_stops_measuring.md`](controller_stops_measuring.md) | Two ways the controller stops measuring while training runs on unaware: a torn read of the files the two processes share (proven, fixed) and the receive loops going quiet (measured, unexplained); plus how to dump greenthreads without root | `utils/atomic_io.py`, `utils/simple_monitor.py`, `utils/greenthread_dump.py`, `loader/train_loader.py` |

For where each manuscript figure and table comes from, see
[`../paper/README.md`](../paper/README.md) instead.
