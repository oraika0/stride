# Negative result: estimated link utilisation, and weighting the loss with it

Removed from the code on 2026-08-19. This is the record of what was tried, so the
idea is not re-derived from scratch by the next person.

## What lu_est was

The reward and the state only ever see what the controller can measure: per-link
throughput, delay and loss. None of that is link *utilisation*, and utilisation is
what traffic engineering actually cares about. The true value needs the traffic
matrix, which an agent running on a real network does not have — using it would
make the setting oracle-assisted rather than partially observed.

`lu_est` was an attempt to reconstruct utilisation from the measurable signals
alone. Two observations drive it:

- **Delay growth.** Under a fluid queue, a link whose arrival rate exceeds its
  service rate accumulates backlog, and the queueing delay grows step over step.
  The growth rate `dD/dt`, converted through the queue's packet size and capacity,
  gives an estimate of by how much arrival exceeds service — that is, of how far
  above 1.0 the utilisation is.
- **Loss.** Once the buffer is full the delay stops growing and the excess shows
  up as drops instead. The loss ratio then estimates the same overshoot.

The two regimes are close to mutually exclusive: delay grows while the queue is
filling, loss appears once it is full and delay has flattened. So the estimator
combined them (sum, effectively a max), gated on the transmit utilisation being
high enough for either signal to mean anything, and clamped the result.

Crucially the estimate is allowed to exceed 1.0. Measured utilisation saturates —
a link at 100% and a link at 300% of capacity look identical in a throughput
counter — and that saturation is exactly what removes the gradient in the
overloaded regime.

## Where it was used

Three places, all off by default and all removed:

1. **As a state channel** (`state_def='lu_est'`) — replace the three measured
   channels of a link token with the single estimated utilisation.
2. **As a reward** (`reward_def='lu_est_sq'`, `'luest_freebw'`, `'global_mlu'`) —
   score a path by the sum of squared estimated utilisation along it, by
   overload-aware free bandwidth `(1 - lu_est)`, or give every pair the same
   global `1/MLU` scalar.
3. **As a per-pair weight on the actor loss** (`use_pw`) — this is the one with
   the cleanest experimental record, below.

## The per-pair weighting experiment

The policy-gradient term is normally an unweighted mean over the N OD pairs:

    actor_loss = -mean_i( rho_i * delta_i * log_pi_i )

`use_pw` multiplied each pair's term by a weight derived from the most congested
link on the path that pair had chosen:

    w_i = alpha * max_{e in chosen path of pair i} lu_est_e

The critic was left unweighted, so only the policy gradient was biased.

The motivation: each pair's reward is min-max normalised, which discards the
absolute level. A saturated pair and an idle pair can end up with similar
normalised rewards, so the gradient budget is spread evenly over pairs that do not
equally need it. The weight was meant to put it back where the congestion is.

Three weightings were run, each differing from the mainline configuration in
nothing but this (same per-pair head, same PMA-2 pooling, same gradient routing,
M = 8, 32-node, seed 17):

| weighting | form | MLU (%) | vs baseline |
| --- | --- | --: | --: |
| none (baseline) | — | 81.02 | — |
| `luest` | `alpha * bottleneck lu_est` | 84.45 | +3.43 |
| `inv_luest_clip` | `clip(alpha / bottleneck lu_est, 0, 1)` | 86.89 | +5.87 |
| `luest_clip` | `clip(alpha * bottleneck lu_est, 0, 1)` | 87.77 | +6.75 |

MLU is maximum link utilisation, lower is better.

All three lost. The first reading was that up-weighting congested pairs amplifies
the noisiest part of training — early on, the pairs with the worst congestion are
the ones whose reward signal is least trustworthy. So the second and third were
designed against that reading: `inv_luest_clip` only ever *reduces* the weight of
saturated pairs and never amplifies anything, and `luest_clip` keeps the original
direction but caps the weight at 1 so the overloaded regime cannot be blown up.

Both corrections did worse than the failure they were correcting. That is the
substantive finding: the problem is not the direction of the weighting or the
absence of a cap. Re-weighting the per-pair policy-gradient terms by observable
congestion does not help here, in any of the three forms tried.

The same conclusion held for the state and reward uses. Replacing the state
channels with `lu_est` failed on the earlier architecture, and putting the global
bottleneck back into the reward was the next attempt after the weighting axis
died; neither reached the unweighted per-pair baseline.

## What is left

Nothing in the code. The fluid-queue parameters `queue_max_pkts`,
`queue_avg_pkt_bytes` and `queue_step_duration` remain in the STRIDE config, but
they belong to the simulator's queue model (`_update_queues`) and are unrelated to
this.

The reward the paper reports is `ls2ic_directed`: per-pair, from measured free
bandwidth, unweighted.
