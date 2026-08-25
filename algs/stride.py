"""STRIDE — diffusion-based routing for SDN traffic engineering.

The routing problem is posed as one joint decision: for each of the N OD pairs,
choose one path out of K frozen candidates. STRIDE does not choose them one at a
time. A discrete-diffusion decoder starts from a fully masked configuration and
refines all N pairs together over M denoise steps, so a pair's choice is made
with the other pairs' tentative choices visible — which is what lets the policy
avoid piling several pairs onto the same link.

Three parts:

    encoder    per-pair link observations -> a fixed-width pair embedding.
               A pair is restricted to the links on its own K candidates by
               confining attention to them, rather than by zeroing the other
               links in the input (`MaskedLinkSelfAttention` + a pooling layer).
    decoder    a transformer over the N pair tokens, run M times, each pass
               unmasking the pairs it is most confident about
               (`StrideDenoiser` / `chain_rollout`).
    RL         actor-critic on the resulting joint action, rewarded by the
               network state the controller measures one period later.

Training runs through the single generic loop in `loader/train_loader.py`, the
same one the per-OD baselines use — it reaches an algorithm only through the
REGISTRY in `algs/__init__.py`, and has no STRIDE-specific branch. That loop
speaks of one "agent" per OD pair and the configs carry `num_agents`, which is
the vocabulary of the earlier per-pair methods, not a description of this one:
STRIDE is a single policy that emits all N choices in one forward pass.

Knobs that exist but are off by default (ELP injection, candidate frames,
transferable heads) are exploratory and are not part of any reported result;
each is marked where it appears.
"""
import math
import copy
import contextlib
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint


class MaskedLinkSelfAttention(nn.Module):
    """Multi-head attention over the links a pair can actually use.

    The mask is what makes this layer specific to the routing formulation: a
    query attends only over the links carried by the pair's own candidate paths.

    `attn_kernel` selects the implementation, not the mathematics. 'manual' is
    the explicit QK^T / softmax / VW chain. 'sdpa' routes the same computation
    through `F.scaled_dot_product_attention`, which on Ampere and later picks a
    memory-efficient kernel in fp32 or flash attention in bf16/fp16 — lower peak
    memory either way, faster in bf16.
    """
    def __init__(self, query_dim, kv_dim, output_dim, embed_dim, num_heads,
                 attn_kernel='manual'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.keyvalue_proj = nn.Linear(kv_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, output_dim)
        if attn_kernel not in ('manual', 'sdpa'):
            raise ValueError(f"attn_kernel must be 'manual' or 'sdpa', got {attn_kernel!r}")
        self.attn_kernel = attn_kernel

    def forward(self, query, keyvalue, key_padding_mask=None):
        # Query and key/value lengths may differ (cross-attention), which is what
        # the PMA pool needs; equal lengths reduce to ordinary self-attention.
        #
        # key_padding_mask is (B, Na, sl_kv) bool, True where the key is a real
        # link for this pair. It is not an optimisation. Without it, attention
        # still pools over every link position in the topology, and the masked-out
        # ones are not inert: they contribute the Linear and attention bias
        # vectors. The pair embedding then comes out nearly identical for every
        # candidate path (measured cross-candidate cosine 0.9999), the policy is
        # uniform, and training never leaves its initial state.
        bs, na, sl_q, _ = query.size()
        _, _, sl_kv, _ = keyvalue.size()
        query    = query.reshape(bs * na, sl_q, -1)
        keyvalue = keyvalue.reshape(bs * na, sl_kv, -1)

        Q = self.query_proj(query)            # (B*N, sl_q, embed)
        KV = self.keyvalue_proj(keyvalue)     # (B*N, sl_kv, 2*embed)
        K, V = torch.chunk(KV, 2, dim=-1)

        Q = Q.reshape(bs * na, sl_q,  self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bs * na, sl_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bs * na, sl_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Build broadcast-compatible attention mask if key_padding_mask provided.
        # Shape: (B*Na, 1, 1, sl_kv) broadcasts over num_heads and sl_q dims.
        # Convention: True = attend (valid), False = skip (padded).
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.reshape(bs * na, 1, 1, sl_kv)

        if self.attn_kernel == 'sdpa':
            # PyTorch 2.0+ optimized attention. Auto-dispatches between
            # flash (fp16/bf16, Ampere+) / mem_efficient (any dtype, Turing+)
            # / math (always). Scaling factor 1/sqrt(head_dim) handled
            # internally. is_causal=False since we want full attention.
            # bool attn_mask: True = attend (matches PyTorch 2.0+ semantics).
            weighted_values = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask, is_causal=False)
        else:
            attn_scores  = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)          # (B*N, H, sl_q, sl_kv)
            weighted_values = torch.matmul(attn_weights, V)         # (B*N, H, sl_q, head_dim)

        weighted_values = weighted_values.transpose(1, 2).contiguous()
        weighted_values = weighted_values.reshape(bs * na, sl_q, -1)

        output = self.out_proj(weighted_values)
        output = output.reshape(bs, na, sl_q, -1)
        return output


class StrideActor(nn.Module):
    """STRIDE policy + value network.

    All 4 sub-modules inlined: encoder (MHA+LN+Linear+GRU) + diffusion
    transformer + per-pair head (einsum-vectorized) + V critic (shared MLP).
    """
    def __init__(self, args, num_link):
        super().__init__()
        # Topology / action dims
        self.K              = args.action_dim                       # 20
        self.num_pairs      = args.num_node * (args.num_node - 1)   # 506 for Geant 23-node
        self.hidden_dim     = args.flat_hidden_dim                  # 32
        self.iter_steps     = args.iter_steps                       # M = 8
        self.mask_idx       = self.K                                # = 20 (mask token at idx K)

        # Diffusion transformer concat layout
        self.d_state        = args.d_state                          # 32
        self.d_action       = args.d_action                         # 12
        self.d_sc           = args.d_sc                             # 12
        self.step_embed_dim = args.step_embed_dim                   # 8
        # What occupies the decision half of the decoder token. 'embed' is the
        # method: an action embedding plus the self-conditioning block.
        # 'tau_only' drops both and is the without-diffusion ablation.
        self.decision_token = getattr(args, 'decision_token', 'embed')
        if self.decision_token not in ('embed', 'tau_only'):
            raise ValueError(f"decision_token must be 'embed'|'tau_only', "
                             f"got {self.decision_token!r}")
        self.d_model = self.d_state + self.d_action + self.d_sc + self.step_embed_dim  # 64

        self.nhead   = args.nhead     # 4
        self.nlayers = args.nlayers   # 2

        self.self_cond_dropout = args.self_cond_dropout  # 0.5

        # Reverse-diffusion posterior, following VQ-Diffusion. noise_schedule is
        # how the mask probability gamma_bar interpolates along reverse time;
        # mask_final is its asymptotic value at t=T. Both only feed the schedule
        # arithmetic: the chain itself always starts fully masked at m=0.
        self.vqd_noise_schedule = getattr(args, 'vqd_noise_schedule', 'linear')
        self.vqd_mask_final     = float(getattr(args, 'vqd_mask_final', 0.9))
        # Initial state of the reverse chain. 'all_mask' is what the method uses.
        # 'stationary' is the textbook VQ-Diffusion x_T (gamma_final mask, the
        # rest uniform over K) and exists for comparison.
        self.vqd_init           = getattr(args, 'vqd_init', 'all_mask')
        # What the decoder feeds back to itself between denoise steps: the soft
        # belief (default), the argmax token, or a sample from it. Detached in
        # every case, and independent of greedy/sampled decoding.
        self.self_cond_mode     = getattr(args, 'self_cond_mode', 'soft')
        # Ablation switch for self-conditioning as an input signal. When off, the
        # feedback block stays zero in both training and evaluation, so d_model is
        # unchanged and only the signal is removed. Setting self_cond_dropout=1.0
        # is not the same thing -- that zeroes it during training only, and leaves
        # evaluation reading a signal the policy never trained against.
        self.use_self_cond      = getattr(args, 'use_self_cond', True)
        # Inference-time decoding. False = greedy argmax at every denoise step,
        # which is what the paper reports; True samples at every step, as training
        # does.
        self.eval_sample = getattr(args, 'eval_sample', False)

        # How the per-link tokens (B, N, num_link, C) collapse into one embedding
        # per pair. 'pma' is a Set-Transformer pooling by attention against
        # learnable seed queries: it treats the links as a set, so it is defined
        # for any link count. 'flatten' concatenates all link slots into one
        # Linear, which is cheaper but fixes the layer to one topology; it is the
        # form the without-encoder ablation uses.
        self.encoder_pool = getattr(args, 'encoder_pool', 'flatten')

        # How the pair embedding h turns into K logits. Each pair owns a scoring
        # matrix, applied to all pairs at once with a single einsum. Candidate
        # identity lives in those weights rather than in a feature, which is what
        # makes it robust -- but the pair count is in the parameter shape, so the
        # head does not carry to a topology with a different one.
        #
        # Heads that score from per-candidate features instead, sharing one scorer
        # across all candidates and all pairs, would transfer. Eleven of them were
        # tried and none learned; see docs/negative_result_transferable_heads.md.

        # Speed/memory toggles that leave the model definition alone. sdpa plus
        # bf16 autocast runs 2-3x faster with lower memory on Ampere and later,
        # and is what every archived run used.
        self.attn_kernel      = getattr(args, 'attn_kernel', 'manual')
        self.mixed_precision  = getattr(args, 'mixed_precision', 'fp32')
        if self.mixed_precision not in ('fp32', 'bf16', 'fp16'):
            raise ValueError(f"mixed_precision must be 'fp32'|'bf16'|'fp16', "
                             f"got {self.mixed_precision!r}")
        # bf16 needs compute capability 8.0. Asking for it on anything older --
        # a V100 is 7.0, and plenty of clusters are still on them -- makes torch
        # raise at the first autocast, several minutes into a run, after the
        # topology and controller are already up. Fall back instead, and say so:
        # fp32 rather than fp16, because fp16's narrower exponent range can
        # overflow where bf16 would not, and this substitution should not be able
        # to make the numbers worse than the request would have.
        if (self.mixed_precision == 'bf16' and torch.cuda.is_available()
                and not torch.cuda.is_bf16_supported()):
            print(f"[stride] this GPU ({torch.cuda.get_device_name(0)}, sm_"
                  f"{''.join(map(str, torch.cuda.get_device_capability(0)))}) has "
                  f"no bfloat16; falling back to fp32. Results may differ slightly "
                  f"from runs made on bf16 hardware.")
            self.mixed_precision = 'fp32'
        self._autocast_dtype  = {'fp32': None,
                                 'bf16': torch.bfloat16,
                                 'fp16': torch.float16}[self.mixed_precision]

        # The encoder lifts each link's raw channels to hidden_dim before any
        # attention, and stays in hidden_dim from there. An earlier version kept
        # the per-link feature at 3 dimensions throughout, which capped the
        # pooled representation's effective rank near 1 and starved everything
        # downstream. Checkpoints do not cross that change.

        # A link token carries three measured channels: bandwidth ratio,
        # normalised delay, and loss percent. Built by
        # loader/train_loader.py:get_state_directed.
        state_fea_dim = 3
        # encoder_spatial is the without-encoder ablation. 'attn' is the method:
        # project, attend within the pair's own links, pool. 'flat' bypasses all
        # three and flattens the masked link state into a single Linear. The GRU
        # stays either way, so what the ablation removes is the spatial encoder,
        # not the temporal memory.
        self.encoder_spatial = getattr(args, 'encoder_spatial', 'attn')
        if self.encoder_spatial not in ('attn', 'flat'):
            raise ValueError(f"encoder_spatial must be 'attn'|'flat', got {self.encoder_spatial!r}")
        # Optional learned identity per link. Link tokens otherwise carry only
        # dynamic measurements, which makes attention and pooling permutation-
        # invariant -- the pair embedding is a summary of an unordered multiset,
        # with link identity absent by construction. This gives each link its own
        # dimensions inside the token, concatenated rather than added: an anchor
        # added into the dynamic dimensions lets the model ride the constant and
        # wash out the measurement, while concatenation keeps the two separable.
        # Concatenating at the channel level instead would be equivalent to
        # adding, so the split has to happen after the projection. Off by default.
        if self.encoder_spatial == 'attn':
            self.input_proj = nn.Linear(state_fea_dim, self.hidden_dim)

        # --- Spatial encoder: 'attn' builds the learned encoder; 'flat' replaces
        # the whole input_proj+attn+pool stack with one Linear (built below). ---
        if self.encoder_spatial == 'attn':
            # Encoder attention now full hidden_dim throughout (no 3-D bottleneck)
            self.encoder_attn = MaskedLinkSelfAttention(
                query_dim=self.hidden_dim, kv_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                embed_dim=self.hidden_dim, num_heads=4,
                attn_kernel=self.attn_kernel)
            self.encoder_ln = nn.LayerNorm(self.hidden_dim)

            # --- Encoder pool: branch on encoder_pool ---
            if self.encoder_pool == 'pma':
                # Number of learnable seed queries in the Set-Transformer pool.
                # Each seed produces its own pooled view of the pair's links; the
                # views are fused back to hidden_dim so nothing downstream changes
                # shape. The reported configuration uses 2.
                self.pma_num_seeds = int(getattr(args, 'pma_num_seeds', 1))
                if self.pma_num_seeds < 1:
                    raise ValueError(f"pma_num_seeds must be >=1, got {self.pma_num_seeds}")
                self.pool_query = nn.Parameter(torch.randn(self.pma_num_seeds, self.hidden_dim) * 0.02)
                self.pool_attn = MaskedLinkSelfAttention(
                    query_dim=self.hidden_dim, kv_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    embed_dim=self.hidden_dim, num_heads=4,
                    attn_kernel=self.attn_kernel)
                if self.pma_num_seeds > 1:
                    # Fuse the k pooled seed vectors back to hidden_dim.
                    self.pool_fuse = nn.Linear(self.pma_num_seeds * self.hidden_dim, self.hidden_dim)
                # NOTE: the old per-link encoder_linear (3 -> hidden_dim) is gone --
                # input_proj at the top already lifts to hidden_dim, encoder_attn
                # output is hidden_dim, so the per-link 3-32 projection became redundant.
            else:  # 'flatten': one Linear over all link slots
                # Attention output is hidden_dim per link, so the flattened
                # width is num_link * hidden_dim.
                self.encoder_linear = nn.Linear(num_link * self.hidden_dim, self.hidden_dim)
        else:  # 'flat': raw (L*state_fea_dim) -> hidden, no attention or pooling.
            # masked link state flattened straight into one Linear. Per-pair
            # identity survives via which link slots are non-zero. Topology-locked
            # (L baked into the weight) -- fine, transfer abandoned.
            self.encoder_flat = nn.Linear(num_link * state_fea_dim, self.hidden_dim)

        self.encoder_gru = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        # --- Diffusion transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=0.0, activation='gelu', batch_first=True)
        self.diffusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.nlayers)

        # --- Action embedding: K real actions + 1 mask token = K+1 entries ---
        if self.decision_token == 'tau_only':
            # Without-diffusion ablation: the token is Linear(h_pair) only, with no
            # action embedding and no self-conditioning. d_model stays 64, so the
            # cross-pair transformer and the head are unchanged; this projection
            # is the only parameter the ablation adds.
            self.tau_only_proj = nn.Linear(self.hidden_dim, self.d_model)
        else:
            self.action_emb = nn.Embedding(self.K + 1, self.d_action)
            nn.init.normal_(self.action_emb.weight, mean=0.0, std=0.02)

        # --- Head: one independent scoring matrix per pair ---
        self.head_W = nn.Parameter(torch.empty(self.num_pairs, self.d_model, self.K))
        self.head_b = nn.Parameter(torch.zeros(self.num_pairs, self.K))
        nn.init.kaiming_uniform_(self.head_W, a=5 ** 0.5)

        # --- V critic: shared 2-layer MLP (NOT per-pair) ---
        v_hidden = args.critic_hidden  # 256
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, v_hidden),
            nn.GELU(),
            nn.Linear(v_hidden, 1),
        )

    def head_parameters(self):
        """Head parameters, which form the 'slow' group of the actor optimizer."""
        return [self.head_W, self.head_b]

    def init_hidden(self, B, N, device):
        """Zero-init GRU hidden state. Used at:
        - Inference start (testing_ma resets agents.hidden_states at line 1268).
        - Training-window start (DRQN A2: zero-init at t=0 inside StrideAgent.update).
        """
        return torch.zeros(B * N, self.hidden_dim, device=device)

    def encoder_to_pool(self, masked_state):
        """Shared MHA + LN + pool body. Used by both encoder_forward (pair branch,
        adds GRU after) and encoder_path_forward (path branch, no GRU).

        Args:
            masked_state: (B, Na, num_link, 3) where Na is N for pair or N*K for paths.

        Returns:
            enc: (B, Na, hidden_dim) -- pre-GRU representation.

        Equivalent to the earlier FlatMaskedEncoder._backbone, with the added
        encoder_pool=='pma' dispatch.
        """
        # 'flat' (without-encoder ablation): bypass input_proj/attn/pool. Raw
        # masked link state (B, Na, L, F) -> flatten (B, Na, L*F) -> one Linear ->
        # (B, Na, hidden). Single fp32 Linear, no autocast/padding-mask needed
        # (zeros are part of the input; encoder_flat learns the active slots).
        if self.encoder_spatial == 'flat':
            B, Na, L, F_ = masked_state.shape
            return self.encoder_flat(masked_state.reshape(B, Na, L * F_))
        # Autocast wrapper when mixed_precision is not fp32. sdpa under bf16 on
        # Ampere and later selects a flash-attention kernel: 2-3x faster and often
        # low enough on memory to skip the gradient checkpoint below.
        if self._autocast_dtype is not None and masked_state.is_cuda:
            ctx = torch.cuda.amp.autocast(dtype=self._autocast_dtype)
        else:
            ctx = contextlib.nullcontext()
        # The padding mask is derived from the input rather than passed in: a link
        # position whose feature vector is all zero is not on any of this pair's
        # candidates. encoder_attn and pool_attn use it to
        # skip masked positions; without them the ~69 of 74 zero positions carry
        # Linear bias vectors that dominate the pool.
        key_padding_mask = (masked_state.abs().sum(dim=-1) > 0)    # (B, Na, L) bool
        with ctx:
            # Lift the link state to hidden_dim before any attention, so the
            # encoder is never bottlenecked by the channel count.
            state_h = self.input_proj(masked_state)            # (B, Na, L, hidden)
            att = self.encoder_attn(state_h, state_h, key_padding_mask=key_padding_mask)
            att = self.encoder_ln(state_h + att)               # residual + LN in hidden_dim

            if self.encoder_pool == 'pma':
                B, Na = att.shape[:2]
                k = self.pma_num_seeds
                query = self.pool_query.view(1, 1, k, -1).expand(B, Na, k, -1)
                enc_pool = self.pool_attn(query, att, key_padding_mask=key_padding_mask)  # (B,Na,k,hidden)
                if k > 1:
                    # PMA_k: flatten the k pooled seed vectors -> fuse back to hidden.
                    enc = self.pool_fuse(enc_pool.reshape(B, Na, k * self.hidden_dim))  # (B,Na,hidden)
                else:
                    enc = enc_pool.squeeze(2)                             # (B, Na, hidden), k=1
            else:  # 'flatten': one Linear over all link slots
                # FC encoder doesn't pool; it flattens. Zero out masked
                # positions explicitly so the Linear sees a clean signal.
                keep = key_padding_mask.unsqueeze(-1).to(att.dtype)
                att = att * keep
                flat = att.reshape(*att.shape[:2], -1)
                enc = self.encoder_linear(flat)
        # Cast back to fp32 for downstream consumers (GRU, value head, etc.
        # were built without explicit dtype handling). Keeps the heavy
        # attention in bf16/fp16, the lightweight tail in fp32.
        return enc.float() if self._autocast_dtype is not None else enc

    def encoder_forward(self, masked_state, h_in):
        """Pair branch: encoder_to_pool + GRU. Returns (h_pair, h_out).

        Args:
            masked_state: (B, N, num_link, 3) -- pair-masked link state.
            h_in:         (B*N, hidden_dim) or None.

        Returns:
            h_pair:   (B, N, hidden_dim) -- stateful (GRU-mixed with prev hidden).
            h_out: (B*N, hidden_dim)  -- carry forward.
        """
        enc = self.encoder_to_pool(masked_state)              # (B, N, hidden)
        enc_flat = enc.reshape(-1, self.hidden_dim)           # (B*N, hidden)
        if h_in is None:
            h_in = torch.zeros_like(enc_flat)
        h_out = self.encoder_gru(enc_flat, h_in)
        return h_out.view(*enc.shape[:2], -1), h_out

    @staticmethod
    def _sinusoidal(t, dim):
        """Standard diffusion-step sinusoidal embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        ang = float(t) * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)])
        if emb.numel() < dim:
            emb = F.pad(emb, (0, dim - emb.numel()))
        return emb

    # ===== VQ-Diffusion: forward schedule + reverse posterior =====
    # The standard VQ-Diffusion reverse step, inlined here rather than imported so
    # the decoder has no cross-algorithm dependency.

    def _vqd_cumulative_schedule(self, t, T):
        """Cumulative forward schedule at diffusion time t (reverse: t goes T -> 1).
        Returns (alpha_bar, gamma_bar, beta_bar):
          alpha_bar = prob token remains original; gamma_bar = prob MASK;
          beta_bar = per-action uniform replace prob."""
        if t <= 0:
            return 1.0, 0.0, 0.0
        if t >= T:
            t = T
        r = float(t) / float(T)
        schedule = self.vqd_noise_schedule
        gamma_final = self.vqd_mask_final
        K = self.K
        if schedule == 'cosine':
            noise_level = 1.0 - math.cos(math.pi * r / 2.0)
        elif schedule == 'sin':
            noise_level = math.sin(math.pi * r / 2.0)
        else:  # 'linear' (default)
            noise_level = r
        gamma_bar = gamma_final * noise_level
        alpha_bar = max(0.0, 1.0 - noise_level)
        beta_bar  = max(0.0, (1.0 - alpha_bar - gamma_bar) / K)
        return alpha_bar, gamma_bar, beta_bar

    def _vqd_per_step_transition(self, t, T):
        """Per-step (alpha_t, beta_t, gamma_t) from consecutive cumulative values.
        alpha_t = stay, gamma_t = -> MASK, beta_t = -> uniform replace."""
        alpha_bar_t,  gamma_bar_t,  _ = self._vqd_cumulative_schedule(t,     T)
        alpha_bar_tm1, gamma_bar_tm1, _ = self._vqd_cumulative_schedule(t - 1, T)
        K = self.K
        alpha_t = (alpha_bar_t / alpha_bar_tm1) if alpha_bar_tm1 > 1e-10 else 0.0
        if (1.0 - gamma_bar_tm1) > 1e-10:
            gamma_t = 1.0 - (1.0 - gamma_bar_t) / (1.0 - gamma_bar_tm1)
        else:
            gamma_t = 0.0
        gamma_t = max(0.0, gamma_t)
        alpha_t = max(0.0, min(1.0, alpha_t))
        beta_t  = max(0.0, (1.0 - alpha_t - gamma_t) / K)
        return alpha_t, beta_t, gamma_t

    def _vqd_compute_reverse_prob(self, x_t, p_x0, t, T):
        """Reverse posterior p(x_{t-1} | x_t). All-log-space for numerical stability.

          p(x_{t-1}=j | x_t) = sum_k p_theta(x_0=k | x_t) * q(x_{t-1}=j | x_t, x_0=k)
          q(x_{t-1}=j | x_t=i, x_0=k) prop_to q(x_t=i | x_{t-1}=j) * q(x_{t-1}=j | x_0=k)

        Args:
            x_t:  (N,) long, states in {0..K-1=action, K=MASK}.
            p_x0: (N, K) float, predicted x_0 probabilities.
            t:    current diffusion time (reverse: T -> 1).
            T:    total reverse steps.
        Returns:
            (N, K+1) reverse transition probabilities for x_{t-1}.
        """
        K = self.K
        MASK = K
        N = x_t.size(0)
        device = x_t.device
        LOG_FLOOR = -70.0

        def _slog(x):
            return math.log(x) if x > 1e-30 else LOG_FLOOR

        alpha_t, beta_t, gamma_t = self._vqd_per_step_transition(t, T)
        alpha_bar_tm1, gamma_bar_tm1, beta_bar_tm1 = self._vqd_cumulative_schedule(t - 1, T)

        log_beta_t   = _slog(beta_t)
        log_gamma_t  = _slog(gamma_t)
        log_at_bt    = _slog(alpha_t + beta_t)
        log_ab_tm1   = _slog(alpha_bar_tm1 + beta_bar_tm1)
        log_bbar_tm1 = _slog(beta_bar_tm1)
        log_gbar_tm1 = _slog(gamma_bar_tm1)

        # log_prior[j, k] = log q(x_{t-1}=j | x_0=k), shape (K+1, K).
        log_prior = torch.full((K + 1, K), log_bbar_tm1, device=device)
        diag_idx  = torch.arange(K, device=device)
        log_prior[diag_idx, diag_idx] = log_ab_tm1
        log_prior[MASK, :]            = log_gbar_tm1

        log_p_x0   = torch.log(p_x0.clamp(min=1e-30)).clamp(min=LOG_FLOOR)        # (N, K)
        is_mask    = (x_t == MASK)
        is_ord     = ~is_mask
        log_reverse = torch.full((N, K + 1), LOG_FLOOR, device=device)

        # Case A: x_t = MASK
        m_idx = is_mask.nonzero(as_tuple=True)[0]
        if m_idx.numel() > 0:
            log_trans = torch.full((K + 1,), log_gamma_t, device=device)
            log_trans[MASK] = 0.0
            log_unnorm = log_trans.unsqueeze(1) + log_prior                       # (K+1, K)
            log_Z      = torch.logsumexp(log_unnorm, dim=0, keepdim=True)         # (1, K)
            log_post   = (log_unnorm - log_Z).clamp(min=LOG_FLOOR)                # (K+1, K)
            log_p_m    = log_p_x0[m_idx].unsqueeze(1)                             # (N_m, 1, K)
            log_post_b = log_post.unsqueeze(0)                                    # (1, K+1, K)
            log_reverse[m_idx] = torch.logsumexp(log_p_m + log_post_b, dim=2)     # (N_m, K+1)

        # Case B: x_t = action i
        o_idx = is_ord.nonzero(as_tuple=True)[0]
        if o_idx.numel() > 0:
            N_o = o_idx.numel()
            ord_x = x_t[o_idx]
            log_trans = torch.full((N_o, K + 1), log_beta_t, device=device)
            log_trans[:, MASK] = LOG_FLOOR                                        # MASK is absorbing
            log_trans[torch.arange(N_o, device=device), ord_x] = log_at_bt
            log_unnorm = log_trans.unsqueeze(2) + log_prior.unsqueeze(0)          # (N_o, K+1, K)
            log_Z      = torch.logsumexp(log_unnorm, dim=1, keepdim=True)         # (N_o, 1, K)
            log_post   = (log_unnorm - log_Z).clamp(min=LOG_FLOOR)                # (N_o, K+1, K)
            log_p_o    = log_p_x0[o_idx].unsqueeze(1)                             # (N_o, 1, K)
            log_reverse[o_idx] = torch.logsumexp(log_post + log_p_o, dim=2)

        reverse_prob = torch.softmax(log_reverse.clamp(min=LOG_FLOOR), dim=-1)
        return reverse_prob

    def chain_rollout(self, h_pair, N, track_step_kdiv=False):
        """M=8 step VQ-Diffusion denoise -> per-pair K-way action.

        Args:
            h_pair:        (B, N, hidden=32) -- the pair representation from
                        encoder_forward. This is the paper's h_i^pair, NOT a
                        temperature.
            N:          int = num_pairs.
            track_step_kdiv: if True, record cross-pair chosen-k entropy at EACH
                        denoise step (B=1, collection only -- shows whether the
                        policy starts diverse and collapses during denoising, or
                        is already collapsed at m=0 / anchor-dominated). Adds 8
                        cheap bincounts; keep False in the update() hot loop.

        Returns dict:
            'final_actions': (N,) numpy int64        -- argmax at last m.
            'logits_last':   (B, N, K) torch float   -- last m's logits.
            'h_last':        (B, N, d_model) torch   -- last m's transformer output.
            'step_kdiv':     list[float] of length M -- per-step entropy/ln(K)
                             (only when track_step_kdiv; else absent).
        """

        B = h_pair.size(0)
        K = self.K
        M = self.iter_steps
        mask_idx = self.mask_idx
        device = h_pair.device

        # Optional per-pair identity anchor: add a fixed
        # learned identity anchor into h_pair (the hidden_dim/identity block) so the
        # cross-pair diffusion transformer has a stable per-pair signal beyond
        # h_pair's content. Policy-side only: the critic's value(h_pair_t) (called
        # OUTSIDE chain_rollout) keeps the un-anchored h_pair -- the value head is a
        # per-pair MLP with no cross-pair attention, so the identity anchor is
        # less relevant there. Added once here -> all M steps see the anchored h_pair.
        # Initial state x_T. self-cond starts zeros.
        # vqd_init is ORTHOGONAL to the sample/greedy (eval_sample) decode knob:
        #   'all_mask'  (default): every position = MASK (deterministic init).
        #   'stationary': ALWAYS sample the true VQD forward stationary
        #               q(x_T) = gamma_final MASK + (1-gamma_final) uniform-over-K,
        #               regardless of train/eval or greedy/sample. (greedy/sample
        #               only controls decoding: action argmax-vs-multinomial and
        #               self_cond softmax-vs-gumbel.) A fresh x_T is drawn per
        #               chain_rollout call = per routing decision, matching VQD's
        #               "sample the prior once per generation".
        if self.vqd_init == 'stationary':
            gamma = self.vqd_mask_final
            u = torch.rand(B, N, device=device)
            rand_tok = torch.randint(0, K, (B, N), device=device)        # uniform over K real
            a_curr = torch.where(u < gamma,
                                 torch.full((B, N), mask_idx, dtype=torch.long, device=device),
                                 rand_tok)
        else:  # 'all_mask'
            a_curr = torch.full((B, N), mask_idx, dtype=torch.long, device=device)
        self_cond = (torch.zeros(B, N, self.d_sc, device=device)
                     if self.decision_token == 'embed' else None)
        soft_prev = None   # last step's soft belief dist; None at m=0

        logits = None
        h = None
        step_kdiv = [] if track_step_kdiv else None
        # per-step mean per-pair top-1 mass of p_x0 (peakedness / multi-modality).
        # High (->1) = peaked policy (argmax self_cond ~ soft); low = multi-modal
        # (self_cond weighted-avg may blend modes). Mirrors step_kdiv (per denoise
        # step), collection-only (B=1).
        step_top1 = [] if track_step_kdiv else None
        # per-step mean per-pair Shannon entropy of p_x0, normalized by ln(K).
        # Full distribution-spread measure (top1 only sees the peak): ->0 = each
        # pair concentrated on one path, ->1 = uniform over K. Different axis from
        # kdiv (cross-pair argmax diversity). Mirrors step_top1, collection-only.
        step_entropy = [] if track_step_kdiv else None
        # Route flapping within one chain: among pairs already committed in two
        # consecutive denoise steps, what fraction changed candidate. A high value
        # means pairs are oscillating against each other instead of settling.
        switch_fracs = [] if track_step_kdiv else None
        # The previous step's reverse posterior, kept so the current step's action
        # embedding can be formed with a straight-through estimator. The hard term
        # is the one-hot of the token actually carried forward and the soft term is
        # the distribution it was drawn from, so the estimator is well-formed and
        # gradient reaches the previous step's logits. None at m=0, where the chain
        # starts fully masked and the embedding is a direct lookup.
        prev_posterior_prob = None
        _lnK = math.log(K) if K > 1 else 1.0
        for m in range(M):
            step_sin = self._sinusoidal(M - m, dim=self.step_embed_dim).to(device)   # (8,)
            step_h   = step_sin.view(1, 1, -1).expand(B, N, -1)                      # (B, N, 8)

            if self.decision_token == 'tau_only':
                # w/o-diffusion: decision token = projected encoder state only.
                # No E_act / self_cond / step. self_cond + soft_prev stay None
                # (their update blocks are gated on decision_token == 'embed').
                x = self.tau_only_proj(h_pair)                                          # (B, N, d_model)
            else:
                # --- embed path: the decision token the method uses ---
                # Concat layout: d_state(32) + d_action(12) + d_sc(12) + step(8) = 64
                # action_h: m=0 looks up MASK_emb directly (no prior posterior to ST
                # from). m>0 uses Gumbel-ST with prev_posterior_prob (over K+1) as the
                # differentiable soft -- the load-bearing gradient channel (self_cond
                # is no_grad auxiliary, Analog Bits standard).
                if prev_posterior_prob is None:
                    # m=0: a_curr = MASK everywhere.
                    action_h = self.action_emb(a_curr)                               # (B, N, d_action)
                else:
                    hard_K1  = F.one_hot(a_curr, num_classes=K + 1).float()          # (B, N, K+1)
                    y_K1     = (hard_K1 - prev_posterior_prob.detach()) + prev_posterior_prob   # ST
                    action_h = torch.einsum('bnj,jd->bnd', y_K1, self.action_emb.weight)  # (B, N, d_action)
                # Self-cond dropout in training: 50% chance reset to zeros
                sc_input = self_cond
                if self.training and torch.rand((), device='cpu').item() < self.self_cond_dropout:
                    sc_input = torch.zeros_like(sc_input)
                x = torch.cat([h_pair, action_h, sc_input, step_h], dim=-1)             # (B, N, 64)
            h = self.diffusion_transformer(x)                                    # (B, N, 64)

            logits = torch.einsum('bnd,ndk->bnk', h, self.head_W) + self.head_b      # (B, N, K)


            # Analog Bits self_cond -- FORWARD-ONLY auxiliary (NO gradient through it).
            # Matches Chen et al. ICLR'23 standard practice and the "not
            # grad path" pattern. self_cond informs the transformer of the model's current
            # belief, but the gradient route is the Gumbel-ST action_h above (which uses
            # the posterior probability), NOT through self_cond.
            # self_cond_mode controls the BELIEF content fed back (orthogonal to
            # the sample/greedy decode knob):
            #   'soft'   (default): Σ_k p_k · emb_k -- probability-weighted avg
            #            embedding. Carries the full distribution but, for a
            #            multi-modal p_x0, the embedding-space average can land
            #            between modes (a "ghost" not matching any real action).
            #   'argmax': emb_{argmax(logits)} -- single committed guess, always a
            #            real action's embedding (no cross-mode blend).
            #   'sample': emb_{sampled token} -- matches the hard token the chain
            #            commits to (train/inference consistent), but noisier.
            # use_self_cond=False (ablation): skip the update entirely so self_cond
            # stays its init zeros in BOTH train and eval (sc block = dead zeros).
            if self.decision_token == 'embed' and self.use_self_cond:
                emb = self.action_emb.weight[:K]                                 # (K, d_action)
                with torch.no_grad():
                    if self.self_cond_mode == 'argmax':
                        a_sc = logits.argmax(dim=-1)                              # (B, N)
                        self_cond = emb[a_sc]                                     # (B, N, d_action)
                    elif self.self_cond_mode == 'sample':
                        probs = F.softmax(logits, dim=-1)
                        a_sc = torch.multinomial(probs.view(-1, K), 1).view(logits.shape[:-1])
                        self_cond = emb[a_sc]
                    else:  # 'soft' (default)
                        if self.training or self.eval_sample:
                            soft_sc = F.gumbel_softmax(logits, hard=False)
                        else:
                            soft_sc = F.softmax(logits, dim=-1)
                        self_cond = torch.einsum('bnk,kd->bnd', soft_sc, emb)

            # VQ-Diffusion reverse step. posterior_prob computed WITH
            # gradient (it's a differentiable function of p_x0 = softmax(logits) via
            # log/logsumexp/softmax). This is what next iteration's Gumbel-ST action_h
            # consumes to backprop into THIS step's logits.
            p_x0 = F.softmax(logits, dim=-1)                                           # (B, N, K), grad-on
            t = M - m                                                                  # reverse time: M -> 1
            if self.decision_token == 'tau_only':
                # w/o-diffusion: NO VQ reverse posterior. policy == softmax(logits).
                # Pad a zero MASK column so the unchanged downstream sampler (which
                # at t==1 slices [:K] and renormalizes) draws a_next ~ softmax(logits)
                # -- identical to the A2C log_pi (log_softmax(logits_last)) -> the IS
                # ratio is exactly 1 in-distribution. iter_steps=1 => t==1 always.
                posterior_prob = F.pad(p_x0, (0, 1), value=0.0)                        # (B, N, K+1), MASK col = 0
            else:
                posterior_prob = self._vqd_compute_reverse_prob(
                    a_curr.view(-1), p_x0.view(-1, K), t, M
                ).view(B, N, K + 1)                                                    # (B, N, K+1), grad-on

            # NaN guard, last line of defence (see get_action's env-
            # boundary guard for the root-cause path). Non-finite posterior rows
            # are replaced with uniform so multinomial survives; LOUD log so a
            # recurring pattern is visible (one-off = env race; recurring at the
            # same chain step = a real numerical bug to dig into).
            if not torch.isfinite(posterior_prob).all():
                bad_rows = ~torch.isfinite(posterior_prob).all(dim=-1)             # (B, N)
                print(f"[stride][WARN] non-finite posterior at chain m={m} "
                      f"({int(bad_rows.sum().item())} pairs) -- replaced with uniform")
                uniform = torch.full_like(posterior_prob, 1.0 / (K + 1))
                posterior_prob = torch.where(bad_rows.unsqueeze(-1), uniform,
                                             torch.nan_to_num(posterior_prob, nan=0.0,
                                                              posinf=0.0, neginf=0.0))

            # Discrete sample of a_next under no_grad (sampling is non-differentiable).
            # stochastic vs greedy mirrors the soft computation above. At t=1, slice off
            # the MASK column for the final action (env can't accept MASK); posterior
            # mass on MASK at t=1 is ~0 by construction (one-hot prior at t=0) so this
            # is mostly safety + renormalize.
            with torch.no_grad():
                stochastic = self.training or self.eval_sample
                if t == 1:
                    pp = posterior_prob[..., :K]                                       # drop MASK column
                    pp = pp / pp.sum(dim=-1, keepdim=True).clamp(min=1e-30)
                    if stochastic:
                        a_next = torch.multinomial(
                            pp.clamp(min=1e-10).view(-1, K), 1
                        ).squeeze(-1).view(B, N)
                    else:
                        a_next = pp.argmax(dim=-1)
                else:
                    if stochastic:
                        a_next = torch.multinomial(
                            posterior_prob.clamp(min=1e-10).view(-1, K + 1), 1
                        ).squeeze(-1).view(B, N)
                    else:
                        a_next = posterior_prob.argmax(dim=-1)
            if switch_fracs is not None:
                # both committed (ord->ord) and candidate changed = within-chain flap
                both_ord = (a_curr != mask_idx) & (a_next != mask_idx)
                n_ord = int(both_ord.sum().item())
                if n_ord > 0:
                    switch_fracs.append(
                        float(((a_curr != a_next) & both_ord).float().sum().item() / n_ord))
            a_curr              = a_next
            prev_posterior_prob = posterior_prob   # save for next iteration's Gumbel-ST action_h

            if step_kdiv is not None:
                # Diversity of the MODEL's clean prediction (p_x0.argmax) at this step,
                # NOT the posterior-sampled a_next (which can be MASK at intermediate t).
                _a_pred = p_x0[0].argmax(dim=-1).detach().cpu().numpy()
                _h = np.bincount(_a_pred, minlength=K).astype(np.float64)
                _p = _h / max(_h.sum(), 1.0)
                _nz = _p[_p > 0]
                step_kdiv.append(float(-(_nz * np.log(_nz)).sum() / _lnK))
                # Mean per-pair top-1 mass: how peaked each pair's p_x0 is (peakedness
                # = 1 - multi-modality). Directly indexes the self_cond blend concern.
                step_top1.append(float(p_x0[0].max(dim=-1).values.mean().item()))
                # Mean per-pair Shannon entropy of p_x0, normalized by ln(K). Full
                # distribution spread (top1 only sees the peak). p_x0 already softmax'd.
                _ent = -(p_x0[0] * p_x0[0].clamp_min(1e-9).log()).sum(dim=-1) / _lnK
                step_entropy.append(float(_ent.mean().item()))

        out = {
            'final_actions': a_curr[0].detach().cpu().numpy().astype(np.int64),
            'logits_last':   logits,
            'h_last':        h,
        }
        if step_kdiv is not None:
            out['step_kdiv'] = step_kdiv
            out['step_top1'] = step_top1
            out['step_entropy'] = step_entropy
        if switch_fracs is not None and switch_fracs:
            out['chain_switch_rate'] = float(np.mean(switch_fracs))
        return out

    def value(self, h_pair):
        """V critic per pair. h_pair: (B, N, hidden) -> (B, N) scalar V."""
        return self.value_head(h_pair).squeeze(-1)


class StrideAgent:
    """ls2ic-style interface: get_action / append_sample / update / update_target /
    save_model / load_model / init_hidden.

    Mirrors algs/ls2ic.py::ls2ic_agent -- train_loader's
    regular-ma catch-all branch in train_loader drives both
    transparently. Stride additionally returns log_pi_beta from get_action for
    A2C-IS importance sampling (ls2ic uses DQN, no log_pi_beta needed).
    """
    def __init__(self, args):
        # Identity / topology
        self.K           = args.action_dim
        self.num_pairs   = args.num_node * (args.num_node - 1)
        self.num_link    = args.num_link            # 74 (Geant directed)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RL knobs
        self.gamma           = args.gamma            # 0.9 (ls2ic default)
        self.rho_clip_max    = args.rho_clip_max     # 1.0
        self.value_loss_coef = args.value_loss_coef  # 0.5
        self.target_tau      = args.target_tau       # 0.005
        self.grad_clip       = args.grad_clip        # 0.5

        # Sequence replay
        self.time_seq       = args.time_seq          # 8
        self.mini_batch_seq = args.mini_batch_seq    # b8 EXP: 8; b4 EXP: 4
        self.batch_size     = self.time_seq * self.mini_batch_seq  # warmup threshold
        self.buffer_size    = args.buffer_size       # 500 (ls2ic default)

        # Head knobs. The reported configuration is per_pair_mlp with the 'both'
        # gradient source.
        self.encoder_rl_grad_src = getattr(args, 'encoder_rl_grad_src', 'both')
        assert self.encoder_rl_grad_src in ('both', 'critic'), \
            f"encoder_rl_grad_src must be 'both' or 'critic', got {self.encoder_rl_grad_src!r}"
        # Candidate-frame + ELP (exploratory; off in reported results).
        self.decision_token = getattr(args, 'decision_token', 'embed')
        # update() window-batching for GPU throughput. C=1 (default)
        # runs the original per-window B=1 loop (byte-identical for every existing
        # run incl. Geant). C>1 batches C windows as B=C in ONE forward (per_pair
        # only) -- math-identical (the (wA+wC)*B/n_w scaling reproduces the looped
        # per-window mean), trades ~C x memory for ~C x fewer/larger kernels. On
        # 32node (992 pairs) the 8 GB card caps C at 2 (~1.75x; profiled).
        self.update_window_chunk = max(1, int(getattr(args, 'update_window_chunk', 1)))


        # Actor + Polyak target (3-step deepcopy + requires_grad=False + eval)
        self.actor = StrideActor(args, num_link=self.num_link).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.target_actor.eval()

        # Replay deque
        self.memory = deque(maxlen=self.buffer_size)

        # Optimizer split: subset-declared slow group +
        # everything-else-fast fallback + sanity assertion. Replaces manual list
        # add, which once silently dropped the PMA pool parameters and left them
        # frozen for a whole run.
        # New submodules added in future default to fast group automatically.
        slow_param_ids = set()
        for p in self.actor.diffusion_transformer.parameters():
            slow_param_ids.add(id(p))
        if getattr(self.actor, 'decision_token', 'embed') == 'embed':
            for p in self.actor.action_emb.parameters():
                slow_param_ids.add(id(p))
        elif self.actor.decision_token == 'tau_only':
            # tau_only (w/o-diffusion): the projection is the actor-side token
            # machinery -> slow group (mirrors action_emb's placement).
            for p in self.actor.tau_only_proj.parameters():
                slow_param_ids.add(id(p))
        # Head params via the actor's helper.
        for p in self.actor.head_parameters():
            slow_param_ids.add(id(p))

        actor_params, encoder_critic_params = [], []
        for p in self.actor.parameters():
            if id(p) in slow_param_ids:
                actor_params.append(p)
            else:
                encoder_critic_params.append(p)

        n_total = sum(1 for _ in self.actor.parameters())
        assert n_total == len(actor_params) + len(encoder_critic_params), (
            f"Param split mismatch: total {n_total} != slow {len(actor_params)} "
            f"+ fast {len(encoder_critic_params)}"
        )

        # Two groups, not three. The paper lists separate learning rates for the
        # actor, the critic and the encoder; here the encoder shares the critic's
        # group because the two rates are equal (3e-4). Splitting them would only
        # matter if they diverged -- but note that changing critic_lr moves the
        # encoder with it.
        self.opt = optim.Adam([
            {'params': encoder_critic_params, 'lr': args.critic_lr},   # critic + encoder
            {'params': actor_params,          'lr': args.lr},          # actor
        ])

        # ls2ic interface compat: train_loader / testing_ma set this directly.
        self.hidden_states = None
        self._update_steps = 0

    def get_action(self, state, epsilon=0.0, **info):
        """Per-step inference, called by train_loader at every Mininet step.

        See spec §5.3 'Interface contract' for state / path_states / link_state_raw distinction.

        Args:
            state:   list-wrapped [input_state]; input_state numpy (N, num_link, 3)
                     PRE-MASKED by train_loader (= link_state_raw x pair_mask).
            epsilon: float -- A2C-IS rarely uses epsilon-explore.
            info:    kwargs from train_loader. For cosine head: contains
                     'path_states' (N, K, num_link, 3) numpy per-path-masked +
                     'link_state_raw' (num_link, 3) numpy raw.

        Returns:
            action:      (N,) numpy int64
            output_info: {'log_pi_beta': (N,) numpy, 'h_last': (B, N, d_model) numpy,
                          'link_state_raw': forwarded from info if cosine head}
        """
        state_for_all_pairs = state[0]   # (N, num_link, 3) numpy, pair-masked
        state_tensor = torch.tensor(state_for_all_pairs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
        # NaN guard, observed to fire with the ELP attention bias enabled:
        # net_info CSV mid-write races can parse to garbage that casts to
        # inf/NaN in fp32 -> h_pair -> softmax(inf-inf)=NaN -> chain-wide poison
        # (and 0*NaN=NaN defeats even zero-init gates). Sanitize at the env
        # boundary and LOG LOUDLY -- if this fires repeatedly the controller
        # pipeline needs the fix, not the policy.
        if not torch.isfinite(state_tensor).all():
            n_bad = int((~torch.isfinite(state_tensor)).sum().item())
            print(f"[stride][WARN] non-finite env state at get_action "
                  f"({n_bad} entries) -- sanitized to 0; check net_info CSV race")
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        self.actor.eval()
        with torch.no_grad():
            h_pair, h_new = self.actor.encoder_forward(state_tensor, self.hidden_states)
            result = self.actor.chain_rollout(h_pair, self.num_pairs, track_step_kdiv=True)

            final_actions = result['final_actions']
            logits_last   = result['logits_last']
            h_last        = result['h_last']

            # Per-pair eps-greedy: each of N pairs independently flips a coin.
            # ~eps fraction of pairs sample uniform random, rest keep chain
            # action. Within a single env step the replay row mixes explore +
            # exploit samples -- avoids the "100% random step OR 100% policy
            # step" lopsidedness of per-call eps-greedy.
            # log_pi_beta uses the proper MIXTURE behavior policy
            # (1-eps)*pi + eps/K (same per-pair formula whether random fired
            # or not, since the behavior policy at each pair IS the mixture).
            # Avoids the TF-Agents EpsilonGreedyPolicy bug (issue #494) where
            # conditional branch log-prob biases IS ratios.
            if epsilon > 0.0:
                eps_mask = np.random.rand(self.num_pairs) < epsilon          # (N,) bool
                if eps_mask.any():
                    random_actions = np.random.randint(0, self.K, size=self.num_pairs)
                    final_actions  = np.where(eps_mask, random_actions, final_actions)

            a_T = torch.tensor(final_actions, dtype=torch.long, device=self.device)
            log_pi_actor = F.log_softmax(logits_last, -1)[0].gather(
                -1, a_T.unsqueeze(-1)).squeeze(-1)                              # (N,)
            if epsilon > 0.0:
                # log_pi_beta(a) = log((1-eps)*pi_actor(a) + eps/K) via logaddexp
                log_policy_term  = float(np.log(1.0 - epsilon)) + log_pi_actor
                log_uniform_term = float(np.log(epsilon / self.K))
                log_pi_beta = torch.logaddexp(
                    log_policy_term,
                    torch.full_like(log_policy_term, log_uniform_term))
            else:
                log_pi_beta = log_pi_actor

        self.hidden_states = h_new.detach()

        output_info = {
            'log_pi_beta': log_pi_beta.cpu().numpy(),
            'h_last':      h_last.cpu().numpy(),
        }
        if 'step_kdiv' in result:
            output_info['step_kdiv'] = result['step_kdiv']   # per-denoise-step entropy
        if 'step_top1' in result:
            output_info['step_top1'] = result['step_top1']   # per-denoise-step top-1 mass
        if 'step_entropy' in result:
            output_info['step_entropy'] = result['step_entropy']   # per-denoise-step p_x0 entropy/lnK
        if 'chain_switch_rate' in result:
            output_info['chain_switch_rate'] = result['chain_switch_rate']   # cf within-chain flap diag

        return final_actions, output_info

    def append_sample(self, info_list, next_state, reward, next_link_state_raw=None):
        """Push delay-aligned transition to replay (ls2ic 2-step delay convention).

        info_list[0] = oldest = a_{t-2} info (has action, log_pi_beta).
        info_list[1] = middle = s_{t-1} info (has input_state; candidate-feature
                                             heads also carry link_state_raw).
        next_state              = current s_t pair-masked state.
        next_link_state_raw     = current s_t raw link state (candidate-feature
                                  heads only, optional kwarg).
        reward                  = r_t (delayed reward from a_{t-2}).

        """
        if True:
            row = {
                'state':       info_list[1]['input_state'],
                'action':      info_list[0]['action'],
                'log_pi_beta': info_list[0]['log_pi_beta'],
                'next_state':  next_state,
                'reward':      reward,
            }
            self.memory.append(row)

    def _sample_window(self, T):
        """Sample mini_batch_seq random T-consecutive windows.

        Direct lift from algs/ls2ic.py::ls2ic_agent.sample_window -- same semantics
        so the logged curves are comparable across stride and ls2ic_dd.
        """
        memory_len = len(self.memory)
        valid_range = memory_len - T + 1
        if valid_range < 1:
            return []
        starts = np.random.randint(0, valid_range, size=self.mini_batch_seq)
        return [[self.memory[i] for i in range(s, s + T)] for s in starts]

    def update(self):
        """Sequence A2C-IS update.

        Per-window backward keeps activation memory bounded (avoids b8 OOM at
        full 512-forward backward graph). Single grad clip + single opt.step.

        DRQN A2: each window's h_online + h_target start from init_hidden(zeros).
        Cross-env-step hidden propagation only happens during inference (see
        get_action), NOT in training (this preserves Markov in the training window).
        """
        if len(self.memory) < self.batch_size:
            return {'buffer_size': len(self.memory)}

        windows = self._sample_window(self.time_seq)
        if not windows:
            return {}

        self.actor.train()
        self.opt.zero_grad()
        n_w = len(windows)

        actor_logs, critic_logs, delta_logs, rho_logs, V_logs = [], [], [], [], []

        # window-batch chunk size. C=1 takes the per-window loop below; C>1 takes
        # the batched loop further down. 32-node sets C=2 via update_window_chunk.
        C = self.update_window_chunk

        for window in (windows if C == 1 else ()):
            # DRQN A2: zero-init both online + target hidden at window start.
            h_online = self.actor.init_hidden(B=1, N=self.num_pairs, device=self.device)
            h_target = self.target_actor.init_hidden(B=1, N=self.num_pairs, device=self.device)

            window_actor, window_critic = [], []

            for t in range(self.time_seq):
                s_t = torch.tensor(window[t]['state'], dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                h_pair_t, h_online = self.actor.encoder_forward(s_t, h_online)

                # encoder_rl_grad_src dispatch.
                h_pair_for_chain = h_pair_t.detach() if self.encoder_rl_grad_src == 'critic' else h_pair_t

                result = self.actor.chain_rollout(h_pair_for_chain, self.num_pairs)
                logits_now = result['logits_last']
                sn = torch.tensor(window[t]['next_state'], dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
                with torch.no_grad():
                    h_pair_t_next, h_target = self.target_actor.encoder_forward(sn, h_target)
                    V_t_next = self.target_actor.value(h_pair_t_next).squeeze(0).detach()

                # log pi_now of stored (collection-time) action
                a_t = torch.tensor(window[t]['action'], dtype=torch.long, device=self.device)
                log_pi_now = F.log_softmax(logits_now, -1)[0].gather(
                    -1, a_t.unsqueeze(-1)).squeeze(-1)

                # IS ratio with V-trace truncation [0, rho_clip_max=1.0]
                log_pi_beta = torch.tensor(
                    window[t]['log_pi_beta'], dtype=torch.float32, device=self.device)
                rho = (log_pi_now - log_pi_beta).exp().clamp(0.0, self.rho_clip_max)

                # V critic on s_t -- always uses non-detached h_pair_t (critic ALWAYS trains encoder)
                V_t = self.actor.value(h_pair_t).squeeze(0)

                r_t = torch.tensor(
                    window[t]['reward'], dtype=torch.float32, device=self.device)
                delta = r_t + self.gamma * V_t_next - V_t

                actor_t  = -(rho.detach() * delta.detach() * log_pi_now).mean()
                critic_t = self.value_loss_coef * (rho.detach() * delta.pow(2)).mean()
                window_actor.append(actor_t)
                window_critic.append(critic_t)

                delta_logs.append(float(delta.mean().item()))
                rho_logs.append(float(rho.mean().item()))
                V_logs.append(float(V_t.mean().item()))

            # Per-window backward; scale by 1/n_w to match .mean() over windows.
            wA = torch.stack(window_actor).mean()
            wC = torch.stack(window_critic).mean()
            ((wA + wC) / n_w).backward()

            actor_logs.append(float(wA.item()))
            critic_logs.append(float(wC.item()))

        # ---- per_pair window-batched path (C>1): C windows stacked as B=C in ONE
        # forward. Math-identical to the loop above; only runs when C>1 (per_pair).
        for ci in (range(0, n_w, C) if C > 1 else ()):
            chunk = windows[ci:ci + C]
            B = len(chunk)
            h_online = self.actor.init_hidden(B=B, N=self.num_pairs, device=self.device)
            h_target = self.target_actor.init_hidden(B=B, N=self.num_pairs, device=self.device)
            chunk_actor, chunk_critic = [], []
            for t in range(self.time_seq):
                s_t = torch.tensor(np.stack([w[t]['state'] for w in chunk]),
                                   dtype=torch.float32, device=self.device)            # (B,N,L,3)
                h_pair_t, h_online = self.actor.encoder_forward(s_t, h_online)            # (B,N,hidden)
                h_pair_for_chain = h_pair_t.detach() if self.encoder_rl_grad_src == 'critic' else h_pair_t
                result     = self.actor.chain_rollout(h_pair_for_chain, self.num_pairs)
                logits_now = result['logits_last']                                    # (B,N,K)
                sn = torch.tensor(np.stack([w[t]['next_state'] for w in chunk]),
                                  dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    h_pair_t_next, h_target = self.target_actor.encoder_forward(sn, h_target)
                    V_t_next = self.target_actor.value(h_pair_t_next).detach()            # (B,N)
                a_t = torch.tensor(np.stack([w[t]['action'] for w in chunk]),
                                   dtype=torch.long, device=self.device)              # (B,N)
                log_pi_now = F.log_softmax(logits_now, -1).gather(
                    -1, a_t.unsqueeze(-1)).squeeze(-1)                                 # (B,N)
                log_pi_beta = torch.tensor(np.stack([w[t]['log_pi_beta'] for w in chunk]),
                                           dtype=torch.float32, device=self.device)
                rho = (log_pi_now - log_pi_beta).exp().clamp(0.0, self.rho_clip_max)   # (B,N)
                V_t = self.actor.value(h_pair_t)                                         # (B,N)
                r_t = torch.tensor(np.stack([w[t]['reward'] for w in chunk]),
                                   dtype=torch.float32, device=self.device)
                delta    = r_t + self.gamma * V_t_next - V_t                           # (B,N)
                actor_t  = -(rho.detach() * delta.detach() * log_pi_now).mean()
                critic_t = self.value_loss_coef * (rho.detach() * delta.pow(2)).mean()
                chunk_actor.append(actor_t)
                chunk_critic.append(critic_t)
                delta_logs.append(float(delta.mean().item()))
                rho_logs.append(float(rho.mean().item()))
                V_logs.append(float(V_t.mean().item()))
            wA = torch.stack(chunk_actor).mean()
            wC = torch.stack(chunk_critic).mean()
            # B/n_w scaling: Σ_chunks (B/n_w)·wA_chunk == mean over the n_w windows
            # of the looped per-window wA (each chunk holds B windows). Exact grad.
            ((wA + wC) * B / n_w).backward()
            actor_logs.append(float(wA.item()))
            critic_logs.append(float(wC.item()))

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt.step()

        self._update_steps += 1
        stats = {
            'policy_loss': float(np.mean(actor_logs)),
            'critic_loss': float(np.mean(critic_logs)),
            'rho_mean':    float(np.mean(rho_logs)),
            'delta_mean':  float(np.mean(delta_logs)),
            'V_mean':      float(np.mean(V_logs)),
            'buffer_size': len(self.memory),
            'update_step': self._update_steps,
        }
        return stats

    def update_target(self):
        """Polyak update entire actor -> target_actor (tau=0.005)."""
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1.0 - self.target_tau).add_(p.data, alpha=self.target_tau)

    def save_ckpt(self, path):
        """Save full state to explicit file path (state_dict format)."""
        torch.save({
            'actor':        self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'opt':          self.opt.state_dict(),
            'update_steps': self._update_steps,
        }, path)

    def load_ckpt(self, path):
        """Load from explicit file path (state_dict format).

        A checkpoint is bound to the architecture that produced it. Loading one
        into a differently-configured agent otherwise fails deep inside
        load_state_dict with a shape error that names a tensor and not the cause,
        so the mismatch is detected here and reported by name.
        """
        ckpt = torch.load(path, map_location=self.device)
        actor_keys = set(ckpt['actor'].keys())

        # The encoder projects each link's channels up to hidden_dim before
        # attending (input_proj), except under encoder_spatial='flat', which
        # replaces the body with encoder_flat. Either key means the checkpoint
        # carries the current encoder. An early version worked at the raw channel
        # width throughout and has neither; its shapes differ across input_proj,
        # encoder_attn, encoder_ln and encoder_linear, so it cannot be loaded. No
        # such checkpoint ships with this repository.
        has_input_proj = ('input_proj.weight' in actor_keys) or ('encoder_flat.weight' in actor_keys)
        if not has_input_proj:
            raise ValueError(
                "Checkpoint predates the widened encoder: it has neither "
                "input_proj.weight nor encoder_flat.weight, so its encoder shapes "
                "do not match this code. It cannot be loaded; retrain.")

        self.actor.load_state_dict(ckpt['actor'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        if 'opt' in ckpt:
            # Tolerated mismatch: the optimizer split divides opt into
            # 2 param_groups (slow actor + fast encoder/critic). Pre-Phase-0
            # ckpts (e.g. jbxfhcgd) saved 1-group opt state -> load_state_dict
            # raises "loaded state dict contains a parameter group that
            # doesn't match the size of optimizer's group". For inference /
            # test we don't need opt state at all; for training resume the
            # user should re-train from scratch on a fresh ckpt.
            try:
                self.opt.load_state_dict(ckpt['opt'])
            except ValueError as e:
                print(f"[stride] WARN: optimizer state load skipped "
                      f"(group count mismatch -- pre-Phase-0 ckpt?): {e}")
        self._update_steps = ckpt.get('update_steps', 0)

    def save_model(self, path):
        """train_loader-facing save (line 478, called with path='./results/stride/model').

        Mirrors ls2ic.save_model directory-path convention but uses state_dict
        format inside (more portable + easier to debug shape mismatch than
        ls2ic's full-pickle torch.save(self.actor, ...)).
        """
        import os
        os.makedirs(path, exist_ok=True)
        self.save_ckpt(f"{path}/stride_ckpt.pt")

    def load_model(self, model_path_dir):
        """testing_ma-facing load (line 1253). Takes directory, reads stride_ckpt.pt."""
        self.load_ckpt(f"{model_path_dir}/stride_ckpt.pt")

    def init_hidden(self, actor, batch_size):
        """ls2ic-convention: testing_ma (line 1268) calls when config['rnn']=True
        to reset inference hidden at start of test session.
        """
        return actor.init_hidden(B=batch_size, N=self.num_pairs, device=self.device)
