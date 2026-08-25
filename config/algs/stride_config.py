"""STRIDE configuration.

Two-layer config: _BASE is the mainline configuration the paper reports, and
each VARIANTS entry lists only what its ablation changes. STRIDE_VARIANT picks
the active variant (default "base"); the seed is --seed on the command line.

"""
import os


_BASE = {
    # --- Identity ---
    "algs_name":              "stride",
    "action_dim":             20,

    # --- Architecture (locked) ---
    "flat_hidden_dim":        32,
    "iter_steps":             8,
    "nlayers":                2,
    "nhead":                  4,
    "d_state":                32,
    "d_action":               12,
    "d_sc":                   12,
    "step_embed_dim":         8,
    "critic_hidden":          256,
    "self_cond_dropout":      0.5,
    # 2026-05-30: VQ-Diffusion proper posterior knobs. Ported from the WEM-era impl.
    # vqd_noise_schedule: how gamma_bar (mask prob) interpolates with reverse time t
    #   in [1, T]. Linear: gamma_bar = mask_final * t/T. cosine/sin tilt where steps
    #   concentrate (cosine -> more refinement near clean end; sin -> near noisy end).
    # vqd_mask_final: gamma_final at t=T -- asymptotic mask prob the forward schedule
    #   targets. NOTE: the chain ALWAYS starts a_curr=all MASK regardless of this
    #   value; vqd_mask_final only feeds the posterior MATH via the schedule.
    "vqd_noise_schedule":     "linear",  # 'linear' (default) | 'cosine' | 'sin'
    "vqd_mask_final":         0.9,       # gamma_final at t=T
    # 2026-05-27: default flipped greedy -> sample. Evidence from v1 real
    # test (b8_sample / b8_sample_dt1 vs b8 / b8_pma): sample-collection
    # variants tested clean under BOTH greedy and sample inference; greedy-
    # collection variants broke under sample inference (OOD). Sample
    # collection injects exploration noise at every step which produces a
    # policy robust to either inference mode. No compute cost difference
    # (gumbel_softmax vs softmax both O(N*K) per chain step).
    "eval_sample":            True,
    "encoder_pool":           "pma",      # PMA pooling with pma_num_seeds vectors
    "pma_num_seeds":          2,          # "PMA-2" in the paper
    "update_window_chunk":    2,
    "seed":                   17,         # override per run with --seed
    "encoder_rl_grad_src":    "both",           # encoder gradient source dispatch;
                                                # 'both' (default) preserves existing b8 behavior,
                                                # 'critic' decouples encoder (b8_pma_critic + b8_pma_sim_critic opt-in)

    "attn_kernel":            "sdpa",           # 'sdpa' | 'manual'
    "mixed_precision":        "bf16",           # 'bf16' | 'fp32' | 'fp16'

    # 2026-05-27 v2 encoder: PMA pool operator. 'attention' = original
    # cross-attention with learnable pool_query; 'mean' = parameter-free
    # mean over link dim (ablation against the cross-attention design,
    # tests whether the second attention is redundant since encoder_attn
    # already does within-pair attention).

    # --- RL A2C-IS (locked) ---
    "rho_clip_max":           1.0,
    "value_loss_coef":        0.5,
    "target_tau":             0.005,
    "softupdate_freq":        1,
    "grad_clip":              0.5,
    "lr":                     1e-4,
    "critic_lr":              3e-4,

    # --- Sequence replay ---
    # 2026-05-27: mini_batch_seq + buffer_size + gamma moved here from
    # individual variants -- they were the same across all variants except
    # b4 (which overrides mini_batch_seq=4). Removing the repetition makes
    # variant defs reflect only their distinguishing features.
    "time_seq":               8,
    "mini_batch_seq":         8,        # b4 overrides to 4
    "buffer_size":            500,
    "gamma":                  0.9,

    # --- Reward (locked; uses ls2ic_dd's path_metrics_to_reward_directed) ---
    "reward_mode":            "all",
    # Reward-component weights (paper eq. 22, Table 7). reward_mode="bwd_only"
    # overrides them to (1, 0, 0).
    "lambda_bwd":             1.0,
    "lambda_delay":           1.0,
    "lambda_pkl":             1.0,

    # Fluid-queue model parameters, read by the simulator's _update_queues.
    # avg_pkt_bytes is the L3 size tc htb rate-limits at (IP 20 + UDP 8 + payload
    # 1460), not the Ethernet frame; step_duration is the controller's monitoring
    # period.
    "queue_max_pkts":         1000,
    "queue_avg_pkt_bytes":    1488,
    "queue_step_duration":    1.0,

    # decision_token: what the decoder token carries besides h_pair and the step
    # embedding. 'embed' is the method: the action embedding plus the
    # self-conditioning block, [h_pair(32) | E_act(12) | self_cond(12) | step(8)].
    # 'tau_only' drops both middle blocks and is the without-diffusion ablation.
    "decision_token":         "embed",

    # --- Link identity embedding, design C (2026-06-12) ----------------------
    # Encoder link tokens carry only dynamic channels -> h_pair is a permutation-
    # invariant MULTISET summary (which-link-is-loaded provably absent).

    # --- Data path (directed pipeline; topology-specific knobs come from the
    #     ENV config, NOT here. 2026-06-04: num_link moved out -- the merge is
    #     {**env, **alg} so a hardcoded num_link here would override the env's.
    #     stride must run on a *_directed env config that declares the DIRECTED
    #     link count (geant_directed=74, 32node_144tm_directed=120). num_node
    #     also comes from env -> num_pairs auto-scales (506 geant / 992 32node). ---
    "state_directed":         True,
    "reward_directed":        True,

    # --- Training ---
    "tm_scale":               3,   # inert in real (only dataset/prepare_dataset.py
                                   # and test_sim_only read it)
    # tm_duration_steps removed 2026-06-04: train_loader recomputes it from the
    # env's tm_duration_training (//10), so the old value (200) was always dead.

    # --- ls2ic interface compat ---
    "rnn":                    True,
    "sim_training":           False,

    # --- Required by train_loader catch-all branch + loop_pairs (verified 2026-05-25) ---
    # train_loader.py:186-188 reads epsilon* without .get() -- must be present.
    # epsilon=0 disables eps-explore (A2C-IS uses Gumbel-ST exploration instead).
    # epsilon_first/second_phase must be non-zero to avoid div-by-zero at lines 504/506
    # (these lines never execute when epsilon=0, but defensive).
    "epsilon":                0.0,
    "epsilon_ini":            0.0,
    "epsilon_final":          0.0,
    "epsilon_first_phase":    1,
    "epsilon_second_phase":   1,
    # loop_pairs (train_loader.py:1759) reads config["use_delta_reward"] without
    # .get() -- KeyError if missing. False = raw reward / 100.0 (ls2ic convention).
    "use_delta_reward":       False,
    "use_global_state":       False,

}


VARIANTS = {
    # Every entry lists ONLY what it changes about _BASE, which is the mainline
    # STRIDE configuration the paper reports. Topology comes from --env and the
    # seed from --seed, so neither appears here: a variant is an
    # architecture, not an architecture-times-topology-times-seed.
    #
    # Variants whose experiments were abandoned were removed on 2026-08-19
    # together with their results; `git log` has them if one is ever needed
    # again. What remains is what the manuscript uses.

    "base":          {},                       # mainline STRIDE (Fig 13-15, 17, Table 8/9)

    # Denoise-step ladder (Fig 18 / Table 10). M=8 is _BASE.
    "M4":            {"iter_steps": 4},
    "M6":            {"iter_steps": 6},
    "M10":           {"iter_steps": 10},
    "M12":           {"iter_steps": 12},

    # Candidate-set size (Table 13). K=20 is _BASE, so there is no "k20" here.
    "k10":           {"action_dim": 10},
    "k15":           {"action_dim": 15},
    "k25":           {"action_dim": 25},
    "k30":           {"action_dim": 30},

    # Component ablations (Fig 19 / Table 11).
    # w/o encoder: flat Linear instead of attention+PMA, AND an all-ones pair
    #   mask so every pair sees the full global link state rather than its
    #   K-path-masked slice. The encoder output is then identical for every
    #   pair and only the per-pair head can tell them apart.
    #   encoder_pool is pinned back to "flatten" because the flat encoder must not
    #   inherit _BASE's PMA pooling — the old variant relied on _BASE defaulting to
    #   "flatten", which it no longer does now that _BASE is the mainline config.
    "flatfc_nomask": {"encoder_spatial": "flat", "encoder_input_global": True,
                      "encoder_pool": "flatten"},
    # w/o diffusion: one decode pass, and no denoise-step conditioning.
    "nodiff":        {"iter_steps": 1, "decision_token": "tau_only"},
    # w/o actor gradient: the encoder is shaped by the critic only.
    "critic":        {"encoder_rl_grad_src": "critic"},
}

ACTIVE_VARIANT = "base"


def build_config():
    variant = os.environ.get("STRIDE_VARIANT", ACTIVE_VARIANT)
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown STRIDE_VARIANT={variant!r}. Available: {list(VARIANTS.keys())}")
    cfg = {**_BASE, **VARIANTS[variant]}

    # 2026-05-27 test-time override: STRIDE_EVAL_SAMPLE=true|false flips eval_sample
    # without needing a separate variant. Lets the same trained ckpt be tested under
    # both greedy + sample inference modes (paper: chain_rollout's gumbel branch on/off).
    # Convention: only applies at test_single_tm/testing_ma; training-time inference
    # collection is governed by the variant's own eval_sample (e.g. b8_sample = True).
    # Seed is a run-level knob, not an architecture one, so it is set here rather
    # than by cloning a variant. Every seed used to need its own VARIANTS entry
    # whose only difference was this number.
    # STRIDE_SEED was replaced by the --seed flag on main.py / test_single_tm.py,
    # which every algorithm honours rather than STRIDE alone -- and which sudo
    # cannot silently drop the way it drops an environment variable. Refuse to
    # run while it is still set, instead of quietly training at seed 17.
    if os.environ.get("STRIDE_SEED", "").strip():
        raise RuntimeError(
            "STRIDE_SEED is no longer read. Pass --seed to main.py / "
            "test_single_tm.py instead, and unset STRIDE_SEED.")

    eval_sample_env = os.environ.get("STRIDE_EVAL_SAMPLE", "").strip()
    if eval_sample_env:
        cfg["eval_sample"] = eval_sample_env.lower() in ('true', '1', 'yes')
        print(f"[stride] STRIDE_EVAL_SAMPLE override: eval_sample={cfg['eval_sample']}")

    # 2026-05-27 performance overrides (orthogonal to STRIDE_VARIANT, so any
    # variant's ckpt can be trained/tested under either kernel/precision combo).
    attn_kernel_env = os.environ.get("STRIDE_ATTN_KERNEL", "").strip().lower()
    if attn_kernel_env in ('manual', 'sdpa'):
        cfg["attn_kernel"] = attn_kernel_env
        print(f"[stride] STRIDE_ATTN_KERNEL override: attn_kernel={cfg['attn_kernel']}")
    elif attn_kernel_env:
        raise ValueError(f"STRIDE_ATTN_KERNEL must be 'manual' or 'sdpa', got {attn_kernel_env!r}")

    precision_env = os.environ.get("STRIDE_PRECISION", "").strip().lower()
    if precision_env in ('fp32', 'bf16', 'fp16'):
        cfg["mixed_precision"] = precision_env
        print(f"[stride] STRIDE_PRECISION override: mixed_precision={cfg['mixed_precision']}")
    elif precision_env:
        raise ValueError(f"STRIDE_PRECISION must be 'fp32'|'bf16'|'fp16', got {precision_env!r}")

    # update_window_chunk is how many windows one update stacks into a single
    # GPU pass. It changes nothing about what training computes -- the gradients
    # are identical -- only parallelism against activation memory, so the right
    # value belongs to the card rather than to the experiment. The default suits
    # 8 GB; a larger card wants more. See README section 5.
    chunk_env = os.environ.get("STRIDE_CHUNK", "").strip()
    if chunk_env:
        try:
            chunk = int(chunk_env)
        except ValueError:
            raise ValueError(f"STRIDE_CHUNK must be a positive integer, got {chunk_env!r}")
        if chunk < 1:
            raise ValueError(f"STRIDE_CHUNK must be >= 1, got {chunk}")
        cfg["update_window_chunk"] = chunk
        print(f"[stride] STRIDE_CHUNK override: update_window_chunk={chunk}")

    cfg["stride_variant"] = variant



    # build_run_archive_dir names the run <_experiment>_<topology>_s<seed>_<time>,
    # so _experiment carries the architecture only.
    cfg["_experiment"] = variant
    print(f"[stride] variant={variant} seed={cfg['seed']}: "
          f"mini_batch_seq={cfg['mini_batch_seq']}, "
          f"buffer_size={cfg['buffer_size']}, gamma={cfg['gamma']}")
    return cfg


config = build_config()
