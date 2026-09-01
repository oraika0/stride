# Manuscript figures and tables — index

Every numbered figure and table in the paper, and exactly what produces it.
Output files are named after their caption, so a generated file drops into the
document without renaming.

Everything here runs offline from `results/` and `dataset/`. Nothing needs the
network or a Weights & Biases account.

## Figures

| # | Generator | Reads | Note |
| --- | --- | --- | --- |
| Fig 13. Training reward on GÉANT | [`figures/reward/make_reward_fig.py`](figures/reward/make_reward_fig.py) | `results/*/train/output.txt`, via [`make_curves_csv.py`](figures/reward/make_curves_csv.py) which the generator calls itself | — |
| Fig 14. Performance evaluation in GÉANT | [`figures/holdout/make_holdout_fig.py`](figures/holdout/make_holdout_fig.py) | `results/*/test/<session>/real/<tm>/` | — |
| Fig 15. Training reward on 32-node | `figures/reward/make_reward_fig.py` | as Fig 13 | — |
| Fig 16. Reward components on 32-node | [`figures/reward/make_reward_components_fig.py`](figures/reward/make_reward_components_fig.py) | same, via `make_curves_csv.py` | [`te_objective_design.md`](../docs/te_objective_design.md) |
| Fig 17. Performance evaluation in 32-node | `figures/holdout/make_holdout_fig.py` | `results/*/test/` | — |
| Fig 18. Denoise-step performance on 32-node | [`figures/denoise_step/make_denoise_step_fig.py`](figures/denoise_step/make_denoise_step_fig.py) | `results/stride/test/` | — |
| Fig 19. Ablation performance on 32-node | [`figures/ablation/make_ablation_fig.py`](figures/ablation/make_ablation_fig.py) | `results/stride/test/` | — |

`figures/holdout/make_holdout_fig.py` emits both topologies in one run: Fig 14
with Table 8, and Fig 17 with Table 9.

## Tables

| # | Generator | Reads | Note |
| --- | --- | --- | --- |
| Table 8. MLU difference on GÉANT | `figures/holdout/make_holdout_fig.py` | same session data as Fig 14 | — |
| Table 9. MLU difference on 32-node | `figures/holdout/make_holdout_fig.py` | same session data as Fig 17 | — |
| Table 10. Denoise-step performance on 32-node | `figures/denoise_step/make_denoise_step_fig.py` | same as Fig 18 | — |
| Table 11. Ablation performance on 32-node | `figures/ablation/make_ablation_fig.py` | same as Fig 19 | — |
| Table 12. Candidate-path ILP and full-graph LP theoretical MLU on 32-node | [`bounds/k_oracle_curve.py`](bounds/k_oracle_curve.py) | `.demands` + `k_paths.json`, plus the unrestricted LP floor from [`bounds/edge_lp_bound.py`](bounds/edge_lp_bound.py) | [`lp_ilp_analysis.md`](../docs/lp_ilp_analysis.md) |
| Table 13. Candidate-Path Sufficiency Analysis on 32-node | [`figures/k_ablation/make_k_fig.py`](figures/k_ablation/make_k_fig.py) | `results/stride/test/` (K=10…30 sessions) | [`lp_ilp_analysis.md`](../docs/lp_ilp_analysis.md) |
| LP / ILP summary (`tables/lp_ilp_summary.md`) | [`tables/build_paper_table.py`](tables/build_paper_table.py) | `.demands` + `k_paths.json` via `bounds/check_min_mlu.py` | [`lp_ilp_analysis.md`](../docs/lp_ilp_analysis.md) |

## Other numbers quoted in the text

| Where | Source |
| --- | --- |
| §V-D inference latency | [`figures/timing/inference/`](figures/timing/inference/) → `timing_inference_bench.md`. Live GPU measurement of one routing decision, ~16 ms. |
| §V-D / §V-G per-step training time | [`figures/timing/train_steps/`](figures/timing/train_steps/) → `timing_train_steps.md`, built by `make_timing_table.py` from `timing_steps.csv` |
| Algorithm box | [`figures/algo/render_algo_stride.py`](figures/algo/render_algo_stride.py) (LaTeX → cropped PNG) |
| Dataset / topology figures (Fig 6-12) | [`figures/dataset/`](figures/dataset/) |
| Fig 2. Control delay timeline | [`figures/control_delay/`](figures/control_delay/) — PowerPoint slide, its PDF export, and `crop_control_delay.py` which crops the export |
| Fig 1, 3, 4, 5 — hand-drawn diagrams | [`figures/diagrams/`](figures/diagrams/) — the manuscript's exports; the draw.io source is **not in this repository** |

## Theoretical bounds — `bounds/`

Mostly not figures themselves: they produce the optimal-MLU reference values the
tables and the K-sufficiency argument are stated against. The exception is
`k_oracle_curve.py`, which writes Table 12 and its CSV beside itself here.

| Script | Answers |
| --- | --- |
| [`bounds/check_min_mlu.py`](bounds/check_min_mlu.py) | minimum MLU per TM — LP relaxation (splitting allowed) or single-path ILP, both restricted to the K candidate paths. Imported by the other two and by `tables/build_paper_table.py`. |
| [`bounds/k_oracle_curve.py`](bounds/k_oracle_curve.py) | optimal MLU as a function of candidate-set size K — how small K can get before the candidate set, not the learner, is the bottleneck |
| [`bounds/edge_lp_bound.py`](bounds/edge_lp_bound.py) | edge-based multicommodity LP over **all** paths with splitting — the absolute floor, with no candidate-set restriction. Closes the "is K=20 enough" question. |

## Shared output settings

Resolution and file formats live in one place, [`figures/_figio.py`](figures/_figio.py):
600 dpi, written as PNG + JPG + SVG + PDF — every plot, whether or not it ends up
in the manuscript. What marks a diagnostic is its filename: manuscript figures are
named after their caption so they drop into the document unrenamed, and everything
else keeps a descriptive lowercase name.

## Notation — paper symbol to code identifier

The thesis names things mathematically and the code names them as identifiers.
This is the bridge. Where a name would be misleading on its own, the note says
why.

### Problem setup (Table 3)

| Paper | Code | Where |
| --- | --- | --- |
| `G = (V, E)`, `C_e` | `bw_r.txt` | one directed link per line, `node1, node2, _, capacity_mbps` |
| `N` | `num_pairs`, `num_agents` | 506 on GÉANT, 992 on 32-node. "agent" is the older per-pair vocabulary; STRIDE decides all N with one policy |
| `K` | `action_dim` | 20 |
| `p_{i,k}` | `k_paths.json` | frozen, order is significant — see [`../dataset/README.md`](../dataset/README.md) |
| `s_t` = `(f_{e,t})` | `state`, `global_state` | built by `get_state_directed` |
| `f_{e,t}` = (bwd, delay, pkl) | the 3 channels of `net_info_directed.csv` | `bwd`, `delay_tc_ms`, `pkloss` |
| `a_t`, `a_{i,t}` | `action` | per-pair candidate index |
| `r_t`, `r_{i,t}` | `agent_reward_list` | per pair, already divided by 100 |
| `γ` | `gamma` | 0.9 |

### Encoder (§4.4)

| Paper | Code | Note |
| --- | --- | --- |
| `b_i` candidate-link mask | `mask` from `get_mask_directed` | applied before the encoder sees the state, hence `masked_state` |
| shared linear lift | `input_proj` | `Linear(3, hidden_dim)` |
| masked self-attention | `encoder_attn` | `MaskedLinkSelfAttention`, restricted by `key_padding_mask` |
| PMA + fuse | `pool_attn` + `pool_fuse` | `pma_num_seeds = 2` |
| shared GRU | `encoder_gru` | after the pool, per §4.4 |
| **`h_i^pair`** | **`h_pair`** | the pair representation. Renamed from `tau` in 2026-08 because `tau_m` in the same function is the temperature |

### Decoder (§4.5)

| Paper | Code | Note |
| --- | --- | --- |
| `M` | `iter_steps` | 8 |
| `m` | `m` | loop index in `chain_rollout` |
| `x_m`, `x_{m,i}` | `a_curr` | current intermediate decision, `K` = MASK |
| MASK | `mask_idx` | index `K`, so the embedding table has `K+1` rows |
| `e_{x_m,i}` | `action_h` ← `action_emb` | |
| `e_{0,i}^m` | `sc_input` ← `self_cond` | zeroed with probability `self_cond_dropout`. Detached in every case; built with `gumbel_softmax` during training and plain `softmax` under greedy evaluation, where the paper describes only the latter |
| `t_sin^m` | `step_h` ← `_sinusoidal` | |
| `z_i^m` | `x` | the concatenation, `d_state + d_action + d_sc + step_embed_dim = 64` |
| `g_i^m` | `h` | output of `diffusion_transformer` |
| `Linear_i` | `head_W`, `head_b` | one matrix per pair, applied with one einsum |
| `l_i^m` | `logits` | |
| `p_i^m` | `p_x0` | predicted clean distribution |
| `ψ_i^m` | `posterior_prob` | from `_vqd_compute_reverse_prob` |
| `Q_m`, `Q̄_m` | `_vqd_cumulative_schedule` | returns `(alpha_bar, gamma_bar, beta_bar)` |
| `γ_M`, `β_M` | `vqd_mask_final`, derived | 0.9, so `β_M` = 0.1 |

### Reward (§4.6)

| Paper | Code | Note |
| --- | --- | --- |
| `bwd_i^path` | `calc_bwd_path` | min over the path's links |
| `delay_i^path` | `calc_delay_path` | sum |
| `pkl_i^path` | `calc_loss_path` | `1 − Π(1 − l)` |
| `ε_delay`, `ε_pkl` | `DELAY_FLOOR_MS`, `LOSS_FLOOR_PCT` | 1.5 ms and 0.001%; the first is the measurement noise floor, not a divide-by-zero guard |
| `r_i^bwd`, `r_i^delay`, `r_i^pkl` | the `[1]` slot of each metric | min-max normalised to **[0, 100]**, not [0, 1] |
| `λ_bwd`, `λ_delay`, `λ_pkl` | `lambda_bwd`, `lambda_delay`, `lambda_pkl` | 1, 1, 1 |
| `r_i` | `reward()` | `loop_pairs` divides by 100 afterwards, which is where the paper's [0, 1] scale comes from |

### Training (§4.7)

| Paper | Code | Note |
| --- | --- | --- |
| `ω`, `θ`, `φ` | encoder / actor / critic parameters | two optimizer groups, not three: `ω` shares `critic_lr` with `φ` |
| `α_θ`, `α_φ` = `α_ω` | `lr`, `critic_lr` | 1e-4, 3e-4 |
| `V_φ(s_t)^i` | `actor.value(h_pair_t)` | the critic lives on the actor module; `value_head` is a shared MLP |
| `V_φ̄(s_{t+1})^i` | `target_actor.value(h_pair_t_next)` | |
| `τ` soft-update | `target_tau` | 0.005 — a different `τ` from the decoder temperature |
| `D` | `self.memory` | `buffer_size` = 500 |
| `π_θ(a_{i,t}|s_t)` | `log_pi_now` | recomputed at update time |
| `π_β(a_{i,t}|s_t)` | `log_pi_beta` | stored at collection time |
| `ρ_{i,t}`, `ρ_max` | `rho`, `rho_clip_max` | 1.0 |
| `y_{i,t}` | `r_t + gamma * V_t_next` | not a named variable |
| `δ_{i,t}` | `delta` | |
| `L_actor`, `L_critic` | `actor_t`, `critic_t` | |
| straight-through | `y_K1 = (hard_K1 − prev_posterior_prob.detach()) + prev_posterior_prob` | |
| `L` sequence length, `bs` | `time_seq`, `mini_batch_seq × time_seq` | 8, and 8 × 8 = 64 |

### Names that mean something else than they look like

| Code | Not what it looks like |
| --- | --- |
| `target_tau` | the Polyak coefficient for the target network, not anything to do with `h_pair` |
| `flat_hidden_dim` | the encoder width, i.e. the dimension of `h_i^pair`. "flat" is left over from a removed encoder, and this must stay equal to `d_state` |
| `iter_steps` | `M`, the number of reverse denoising steps |
| `action_dim` | `K`, the number of candidate paths |
| `num_agents` | `N`, the number of OD pairs |
| `decision_token='tau_only'` | the without-diffusion ablation, not anything to do with temperature |
| `sim_training` | simulator-based **training**, which is unmaintained. Simulated *scoring* during a test is a different thing |
