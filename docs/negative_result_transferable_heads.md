# Negative result: transferable routing heads

## Why transfer was the point

The architecture is aimed at generalisation, and generalisation here is a
spectrum, not a single property. At the near end: the same topology under traffic
the policy has not seen. In the middle: operating from measurements alone, with no
traffic-matrix oracle at decision time, which is the setting the whole measurement
pipeline exists to support. At the far end: **a topology the policy was never
trained on.**

Cross-topology transfer is the most demanding point on that spectrum, and it is
the one that decides whether the others are worth much. A policy that generalises
across traffic but has to be retrained from scratch for every network is a
per-network artefact — the training cost recurs every time it is deployed
anywhere new, and nothing learned about routing carries over. A policy that
transfers is a routing model. That is the difference the design was reaching for,
and it is why transfer shaped the architecture rather than being bolted on at the
end.

So every component was chosen to be topology-agnostic:

| component | why it does not depend on the topology |
| --- | --- |
| link encoder | attends over the pair's links and pools them as a **set**, so it is defined for any link count |
| candidate rank | K is the k-shortest-path index — "the 3rd shortest path" means the same thing on any graph |
| decoder | a transformer over N pair tokens; attention has no fixed N |
| critic | pooled per pair, applied identically to each |

Nothing in that list has the number of OD pairs or the number of links baked into a
parameter shape. Weights trained on one graph could be loaded on another.

Except the head. That is what this note is about.

## Where the head sits

The decoder produces one vector per pair:

```
h : (B, N, 64)        N = OD pairs — 506 on Geant, 992 on 32-node
```

The head turns it into one logit per candidate:

```
logits : (B, N, K)    K = 20
```

The encoder can also produce a feature per candidate, if the head asks for it:

```
path_feat : (B, N, K, 32)
```

`path_feat[b, n, k]` is candidate k of pair n, encoded from the link states along
that route. It is not free — computing it is a second pass through the encoder over
N·K = 19 840 instances on 32-node, which is why `uses_path_feat` gates a whole
parallel code path: the per-candidate encoder pass, extra replay-buffer columns, and
the static candidate masks.

## The head that works, and the one thing wrong with it

`per_pair_mlp` ignores `path_feat` completely:

```python
head_W : (N, 64, K)      # 992 × 64 × 20 = 1.27 M parameters on 32-node
head_b : (N, K)
logits = einsum('bnd,ndk->bnk', h, head_W) + head_b
```

Every pair owns a 64×K matrix of its own. What "candidate 7" means for pair 3 is
whatever `head_W[3, :, 7]` learned it to mean, independently of `head_W[500, :, 7]`.
Candidate identity lives in a **parameter**, not in a feature.

This works, and it is easy to see why. The head is handed, per pair, a private set
of K weight vectors, so it can memorise which of that pair's candidates is good
under which decoder state. It never has to recognise anything.

It is also, and for the same reason, the one component that cannot move. `N` is in
the parameter shape. A different topology has a different pair count and the tensor
does not fit; even the same topology with the pairs renumbered would scramble it.
The architecture is transferable everywhere except its last layer, which is the
same as not being transferable.

So the target was a head with **no N in any parameter shape**: one scorer applied
identically to every candidate of every pair, where the candidate's identity arrives
through `path_feat` instead of through weights. That head would load unchanged on a
new graph, and the transfer claim would be real.

## The eleven attempts

All are siamese over both K and N — one set of weights, applied per candidate.

| head | how the logit is formed | shapes |
| --- | --- | --- |
| `cosine` | CLIP-style: L2-normalise both sides, scale by a learnable temperature | `score_proj: 32→64`; `logit_k = τ·⟨ĥ, p̂_k⟩` |
| `dot` | the same without normalising, fixed `1/√64` scale — keeps the load magnitude that L2 discards | `score_proj: 32→64` |
| `mlp` | score the concatenation, so h and the candidate interact nonlinearly | `Linear(64+32 → 64) → GELU → Linear(64 → 1)` |
| `mlp_concat` | `mlp` plus a rank embedding concatenated in its own dimensions | `Linear(64+32+12 → 64) → … → 1` |
| `hypernet` | generate a per-pair projection from the pair's own candidate set, then score bilinearly — a transferable analogue of `per_pair_mlp` | `g: 32→64→(64·32)`; `W_i = g(mean_k path_feat_i)`; `logit_k = hᵀ W_i p_k` |
| `hypernet_concat` | `hypernet` with rank concatenated into the candidate side | `g` output `64·(32+12)` |
| `cosine_mixpair` | fuse the pair token into the candidate key first, so the candidate side is pair-aware | `Linear(32+32 → 64) → GELU → Linear(64 → 64)` |
| `attn_head_cat` | a transformer over K+1 tokens (one h token, K candidate tokens); each token is `content(48) ‖ type(4) ‖ rank(12)` = 64 | 1 layer, 4 heads, shared scorer per candidate token |
| `attn_head_nocat` | the same with the rank block removed and content widened to 60 | `content(60) ‖ type(4)` = 64 |
| `shared_concat_k` | flatten all K candidates into one input and emit all K logits at once | `Linear(64 + 20·32 = 704 → 256 → 64 → 20)` |
| `shared_mlp_k` | `per_pair_mlp`'s shape with the weights shared across pairs — ignores `path_feat` | `h(64) → K` |

The list is not arbitrary. It walks the space of ways two vectors can be compared —
normalised similarity, unnormalised similarity, learned nonlinear interaction,
bilinear with a generated matrix, and full attention — plus the two
information-theoretic corners (`shared_mlp_k` reads no candidate feature at all;
`shared_concat_k` reads all of them at once).

## Results

Geant, same encoder and decoder throughout, only the head differs. MLU, lower is
better; ILP optimum ≈ 0.67, shortest path ≈ 0.87, so 0.95 and above is worse than
not learning.

**Round 1 — the sweep.**

| head | MLU | |
| --- | --: | --- |
| `per_pair_mlp` | **0.649** | the only one that learns |
| `dot`, `mlp`, `hypernet_concat`, `shared_mlp_k`, `cosine` | 0.95 – 1.00 | all dead |
| `attn_head_cat` | ~1.0 | action entropy 0.37 — see below |

**Round 2 — two repairs, both failed.**

| change | MLU | what happened |
| --- | --: | --- |
| `attn_head_nocat` — drop the rank block, widen content to 60 | 0.99 | with the rank gone the transformer has no positional signal at all; the policy is uniform |
| max-pooling in the encoder's candidate pass, to surface the bottleneck link | 0.99 | backfired; cross-candidate cosine rose 0.81 → 0.95 |
| both together | 0.98 | |

**Round 3 — a clean re-test after the decoder was rebuilt.**

| head | MLU | |
| --- | --: | --- |
| `hypernet` | 1.000 | cross-candidate cosine 0.978 |

## Why they all fail, and why it is not the head's fault

A shared scorer can only rank candidates it can tell apart. The diagnostic is the
**cross-candidate cosine**: the average similarity between `path_feat[n, k]` and
`path_feat[n, k']` within a pair. Near 1 means the K candidates of a pair are the
same vector, and then no scorer without per-pair parameters can output anything but
a uniform policy — regardless of how the comparison is done. That is why eleven
different comparison mechanisms produce one result.

It is near 1, for two independent reasons.

**Pooling dilutes the bottleneck.** A candidate's feature is pooled over the links
along its route. Two candidates of the same pair share most of their links — they
are the k-th and k'-th shortest paths between the same endpoints — and pooling
soft-averages, so the one congested link that distinguishes them is averaged among
five or six idle ones.

The obvious fix, max-pooling instead of averaging, makes it worse: two candidates
that share the *bottleneck* and differ elsewhere both take their maximum from that
same link, so max-pooling pulls them together rather than apart. Measured 0.81 →
0.95.

**Masking, or the lack of it.** Attention pooled over every link position in the
topology, not only the pair's own. The masked-out positions are not inert — they
still contribute the Linear and attention bias vectors — so with about 69 of 74
positions masked, nearly the whole pooled vector was bias, identical for every
candidate. Measured cross-candidate cosine 0.9999, policy exactly uniform, training
never left its initial state. Adding `key_padding_mask` fixed that particular
defect, and the fix is in the shipped code because the mask is correct regardless of
head. It was not sufficient: the pooling dilution above remains.

**Why `per_pair_mlp` is immune.** It never reads `path_feat`. Candidates are told
apart by `head_W[n, :, k]` versus `head_W[n, :, k']`, different parameters by
construction. The features can collapse entirely and the head still routes — which
is exactly the property that makes it untransferable. The same per-pair table that
survives feature collapse is the table that cannot be carried to another graph.

## The rescue attempt, and the shape of its failure

If candidates encode identically, break the symmetry by hand: add a learnable
vector per candidate **rank** after pooling. Rank is the k-shortest-path index, so
an anchor indexed by rank stays topology-independent — unlike a per-link embedding,
which would tie the model to one graph and defeat the purpose. A learnable scale on
the anchor, regularised toward zero, was meant to let it fade as the encoder learned
to separate candidates from link state on its own.

`attn_head_cat` shows what happened instead. Given a 12-dimensional rank block, the
transformer learned to score from the rank and ignore the content — action entropy
0.37, a near-deterministic preference for particular ranks regardless of network
state. It is a shortcut, and a reasonable one: rank predicts quality on average,
because shorter paths really are usually better. The model rides the anchor and
never learns to read the state.

Remove the anchor and you get `attn_head_nocat`: nothing separates the candidates,
and the policy is uniform. Both ends fail, symmetrically. With a stable per-rank
signal available the model uses only that; without it there is nothing to use.

This is also why the link identity embedding, added to the encoder later, is
**concatenated** into its own dimensions rather than summed into the dynamic ones.
An anchor added into the signal dimensions is one the model can ride while washing
out the measurement; concatenation keeps the two separable.

## What this costs the paper, and what would be needed

The far end of the spectrum is gone. What remains is the rest of it —
generalisation to unseen traffic matrices, operation from measurements with no
traffic-matrix oracle, joint decisions across pairs — evaluated on each topology
separately, with a head trained for that topology. Those were never merely a means
to transfer; they are what makes the policy deployable at all. But the model is a
per-network artefact, and the ambition that shaped the architecture is not what the
results support.

The eleven heads were swept over their obvious axes: similarity versus MLP versus
bilinear versus attention, rank concatenated or not, mean- or max-pooled encoder.
The failure is upstream of all of them, in how distinguishable the candidate
features are, so escaping it needs a different mechanism rather than a twelfth
scorer:

- **distillation** — train a shared head as a student against `per_pair_mlp` as
  teacher, so the student learns from a working policy rather than from the reward
  directly, where the collapse blocks it from ever getting a useful gradient;
- **a tripartite encoder** — keep pair, candidate and link as separate node types
  rather than pooling links into one candidate vector, so a candidate's
  distinguishing link is never averaged away in the first place;
- **report per-topology results and drop the claim** — which is what the paper does.
