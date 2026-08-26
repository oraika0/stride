# 重要性取樣比與控制延遲

STRIDE 的 off-policy A2C 用重要性取樣比 $\rho$ 修正 behavior 與 target policy 的落差。
而在這個系統裡，$\rho$ 的**分子與分母條件在不同的 state**，兩者差一個監測週期。這是控制
延遲的必然結果，不是實作疏漏。這篇說明它為什麼成立、以及誤差被什麼吸收。

相關：[`../AGENTS.md`](../AGENTS.md) 的 control delay 段、
[`../utils/simple_monitor.py`](../utils/simple_monitor.py) 的監測週期迴圈。

---

## 1. 實際算的比值

$$\rho_{t}=\frac{\pi_\theta(a_{t-2}\mid s_{t-1})}{\pi_\beta(a_{t-2}\mid s_{t-2})}$$

這是 2-step 延遲慣例（$t_d = 1$）下的形式。一般式為 $a_{t-t_d-1}$ 與 $s_{t-t_d-1}$。

**分子**在 update 時重算，用的是 transition 裡存的 state $s_{t-1}$。

- [`algs/stride.py:1062`](../algs/stride.py) 取 `window[t]['state']`（即 $s_{t-1}$）並
  encode，[`:1079`](../algs/stride.py) gather 出 $a_{t-2}$ 的 $\log \pi_\theta$

**分母**不是重算的，是**動作生成當下**就存下來的，因此條件在 $s_{t-2}$。

- [`:1083`](../algs/stride.py) 讀 `window[t]['log_pi_beta']`，該值來自 `append_sample`
  的 `info_list[0]`（[`:1008`](../algs/stride.py)），也就是 $a_{t-2}$ 那一筆 bundle，
  而 $a_{t-2}$ 是由 $s_{t-2}$ 生成的
- 產生它的地方在 [`:951`–`:978`](../algs/stride.py)，取的是 $\epsilon$-mixture
  behavior policy 而非純 actor 機率

$\rho$ 有截斷。[`:1085`](../algs/stride.py) 的
`(log_pi_now - log_pi_beta).exp().clamp(0, rho_clip_max)`，$\bar\rho = 1.0$，即 V-trace
truncation。

分子分母的 state 差一格。下一節說明為什麼必然如此。

## 2. 為什麼差一格

$a_{t-2}$ 生成於週期 $t-2$（輸入 $s_{t-2}$），經控制延遲於週期 $t-1$ 才生效，驅動
$s_{t-1} \to s_t$，而 reward $r_t$ 於週期 $t$ 量到。於是兩個用途要的 state 天生不同。

| 用途 | 要的 state | 理由 |
| --- | --- | --- |
| critic 與 advantage | $s_{t-1}$，生效態 | $\delta = r_t + \gamma V(s_t) - V(s_{t-1})$，前態必須是 $a_{t-2}$ 生效當下的 $s_{t-1}$（[`:1092`](../algs/stride.py)） |
| policy 的 $\pi_\beta$ | $s_{t-2}$，生成態 | behavior 機率只能是 $a_{t-2}$ 被抽出來那一刻的，即條件在 $s_{t-2}$ |

「effective-MDP 的理想分母」會是 $\pi_\beta(a_{t-2} \mid s_{t-1})$。但 **behavior policy
從未在 $s_{t-1}$ 上評估過 $a_{t-2}$** —— 它是在 $s_{t-2}$ 抽的。那個量不存在，取不到。
實作只能用生成當下的 $\pi_\beta(\cdot \mid s_{t-2})$ 頂替。

## 3. 為什麼可接受

### 3a. $s_{t-1}$ 與 $s_{t-2}$ 是同一 demand 的兩次下游觀測

這個問題是 spatial POMDP。agent 觀測到的是 link state，那是 traffic demand 的**下游
效應**，真正的 demand（TM）是 latent 的。GRU encoder 把觀測歷史聚合成對 latent demand 的
belief（$h^{\text{pair}}$），policy 實際作用在這個 belief 上，不是 raw link state。

短窗內 demand 視為同一個 epoch、近似常數。$s_{t-2}$ 與 $s_{t-1}$ 因此只是**同一個 demand
的兩次下游觀測**，差別在量測時點。經 GRU 聚合後兩者反映同一個 demand，在 belief 空間相似。
於是

$$\pi_\beta(\cdot\mid s_{t-2}) \approx \pi_\beta(\cdot\mid s_{t-1})$$

$\rho$ 就近似乾淨的 effective-MDP ratio。誤差量級是
$O(\lVert \text{belief}(s_{t-1}) - \text{belief}(s_{t-2}) \rVert)$，在 demand 緩變下小。

**這不是額外假設。** 它就是「用 GRU 近似 POMDP belief、讓延遲系統可當 Markov MDP 處理」
這整套修正的同一個地基。

### 3b. 截斷與 detach 吸收殘差，梯度方向不受影響

$\rho$ 是 **detach 的權重**，而且截斷到 $[0, \bar\rho]$，$\bar\rho = 1$
（[`:1094`–`:1095`](../algs/stride.py)）。V-trace 本來就是拿偏差換變異的偏估計，3a 的
近似殘差只是再加一點被截斷吃掉的偏差。

**梯度方向不受它影響。** 梯度來自 $\nabla \log \pi_\theta(a_{t-2} \mid s_{t-1})$，那是在
effective state $s_{t-1}$ 上算的，方向正確 —— 增強在生效態 $s_{t-1}$ 表現好的 $a_{t-2}$。

一句話。state 錯位被「demand 緩變使 belief 相似」與「V-trace 截斷」兩層吸收，兩者與
delay-corrected MDP 是同一組假設，不是額外的破綻。

## 4. 論文寫的是 index 對齊的抽象形式

文獻寫 behavior policy 一律用**完整條件、且與 tuple 的 state 同 index**。ACER 存
$\mu(\cdot \mid x_t)$，IMPALA 的 V-trace 寫 $\pi(a_t \mid x_t) / \mu(a_t \mid x_t)$。
沒有任何重要性取樣的論文在寫出來的 tuple 裡放錯位的條件 —— 延遲在資料管線裡就吃掉了，
寫出來是乾淨的 index 對齊。

STRIDE 的 tuple 與 IS 比也照這個慣例寫。

```
(s_t, a_t, r_t, s_{t+1}, {π_β(a_{i,t}|s_t)}_{i∈I})
ρ_{i,t} = π_θ(a_{i,t}|s_t) / π_β(a_{i,t}|s_t)
```

延遲收進「delay-corrected」一詞。本文 §1–§3 記錄的是那個抽象底下、實作真正在算的東西。
