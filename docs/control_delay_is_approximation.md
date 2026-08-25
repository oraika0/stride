# Control-Delay IS Approximation

> STRIDE 的 off-policy A2C 重要性取樣比 ρ 的分子分母條件在**不同 state**(差一個監測週期),這是 control delay 的必然結果。本文記錄「為什麼可接受」的論證,作為口委追 code 時的口頭防禦。**論文本文不攤這個**——tuple 一律用 index 對齊的 abstraction 寫(見最後一節)。

相關:[`../AGENTS.md`](../AGENTS.md) 的 control delay 段、[`../utils/simple_monitor.py`](../utils/simple_monitor.py) 的監測週期迴圈。

---

## 1. 實際算的 ratio(對 code 確認)

$$\rho_{t}=\frac{\pi_\theta(a_{t-2}\mid s_{t-1})}{\pi_\beta(a_{t-2}\mid s_{t-2})}$$

(以 `ls2ic` 2-step delay convention、t_d=1 為例;一般式為 `a_{t-t_d-1}` / `s_{t-t_d-1}`。)

- **分子** π_θ 在 update 時用 tuple 的 state **s_{t-1}** 重算
  - `algs/stride.py:2044` encode `window[t]['state']`(= s_{t-1})→ `:2063` gather 出 a_{t-2} 的 log π_θ
- **分母** π_β 是**生成當下**存的、條件在 **s_{t-2}**
  - `algs/stride.py:2068` 取 `window[t]['log_pi_beta']`,該值源自 `append_sample` 的 `info_list[0]`(= a_{t-2} 那筆 bundle,`stride.py:1930`),而 a_{t-2} 是由 s_{t-2} 生成的
- ρ 截斷:`stride.py:2069` `(log_pi_now - log_pi_beta).exp().clamp(0, rho_clip_max=1.0)`(V-trace truncation)

分子分母 state 差一格,**不是 bug**,見下節。

## 2. 為什麼會差一格(control delay 的必然)

a_{t-2} 生成於週期 t-2(輸入 s_{t-2}),經 control delay 於週期 t-1 才生效,驅動 s_{t-1}→s_t,reward r_t 於週期 t 量到。於是兩個用途要的 state 天生不同:

| 用途 | 要的 state | 理由 |
|---|---|---|
| critic / advantage | **s_{t-1}**(生效態) | δ = r_t + γV(s_t) − V(s_{t-1}),前態是 a_{t-2} 生效當下的 s_{t-1}(`stride.py:2076`)✓ |
| policy 的 π_β | **s_{t-2}**(生成態) | behavior prob 只能是 a_{t-2} 被抽出來那一刻的,即條件在 s_{t-2} |

「effective-MDP 的理想分母」是 π_β(a_{t-2}∣**s_{t-1}**)——但 **behavior policy 從未在 s_{t-1} 上評估過 a_{t-2}**(它是在 s_{t-2} 抽的),這個量不存在。code 只能拿生成時的 π_β(·∣s_{t-2}) 頂替。

## 3. 為什麼可接受 — 近似論證

### 3a. POMDP / GRU:s_{t-1} 與 s_{t-2} 是同一 demand 的不同下游觀測

本問題是 spatial POMDP:agent 觀測 link state(demand 的**下游效應**)去推 latent 的 traffic demand(TM)。GRU encoder 把觀測歷史聚合成對 latent demand 的 belief(`h^pair`),policy 實際作用在這個 belief 上、不是 raw link state。

短窗內 demand 視為同一 epoch(~常數),s_{t-2} 與 s_{t-1} 只是**同一個 demand 的兩次下游觀測**(差在量測時點),經 GRU 聚合後得到的 belief 反映同一 demand → 兩者在 belief 空間相似。因此

$$\pi_\beta(\cdot\mid s_{t-2}) \approx \pi_\beta(\cdot\mid s_{t-1}),$$

ρ 就 ≈ 乾淨的 effective-MDP ratio。誤差量級 O(‖belief(s_{t-1}) − belief(s_{t-2})‖),在 demand 緩變下小。**這不是額外假設**:它就是「GRU 把 POMDP belief 近似掉、讓延遲系統可當 Markov MDP」整套修正的同一個地基。

### 3b. V-trace 截斷 + detach:殘差被吸收,梯度方向不受影響

ρ 是 **detach 的權重**、且截斷到 [0, ρ̄=1]。V-trace 本來就是「拿偏差換變異」的偏估計;3a 的近似殘差只是再加一點被截斷吃掉的偏差。**梯度方向不受它影響**——梯度來自 ∇log π_θ(a_{t-2}∣s_{t-1}),在 effective state s_{t-1} 上是對的(reinforce 在生效態 s_{t-1} 表現好的 a_{t-2})。

> 一句話:state 錯位被「demand 緩變 → belief 相似(GRU/POMDP)」+「V-trace 截斷」兩層吸收,與 delay-corrected MDP 同假設,不是新破綻。

## 4. 論文本文怎麼寫(別攤這個)

文獻(ACER 存 `μ(·|x_t)`、IMPALA V-trace `π(a_t|x_t)/μ(a_t|x_t)`)寫 behavior policy 都用**完整 condition、且與 tuple 的 state 同 index**;沒有任何 IS paper 在寫出來的 tuple 裡放錯位 condition。delay 在資料管線吃掉,寫出來是乾淨 index 對齊。

故 STRIDE tuple 與 IS 比一律寫 index 對齊的 abstraction(§3.4 / §4.7 / 演算法現行):

```
(s_t, a_t, r_t, s_{t+1}, {π_β(a_{i,t}|s_t)}_{i∈I})        # = ACER 的 (x_t, a_t, r_t, μ(·|x_t))
ρ_{i,t} = π_θ(a_{i,t}|s_t) / π_β(a_{i,t}|s_t)             # 全在 s_t,乾淨
```

本文(§3)的錯位**不進論文**;abstraction 把 delay 收進「delay-corrected」一詞即可。本 doc 僅作口委追 code 時的防禦稿。
