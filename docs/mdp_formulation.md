# Problem Formulation

> 本文件記錄 SDN routing RL formulation 的設計動機、結構假設、與訓練設計考量。

## 1. 起點：TM 不可觀測，policy 只能依賴 link-observation

真實問題的資料生成過程由流量矩陣（TM）驅動：TM 決定每個 OD pair 的需求，而 routing policy 的職責是把這些需求妥善分配到候選路徑上。**問題的關鍵在於 TM 本身不可直接觀測**——SDN controller 唯一能取得的資訊是 per-link 量測（頻寬、延遲、封包遺失），這些量是 TM 在當前 action 下所誘導出的下游觀測：

$$
\text{link-obs} = f(\text{TM},\ \text{action},\ \text{network dynamics})
$$

此即一個 **partial observation** 設定：底層真實狀態（TM、queue、flow status）對 policy 不可見，policy 唯一可利用的訊號是這個下游觀測（以及由它衍生出的 reward）。下一節我們論證為何此觀測約束自然導向 link-observation 空間上的 MDP。

## 2. 從 MAB 到 link-observation MDP 的層級遞進

我們把環境建模的選擇拆成幾個層級遞進，每一層對應實際 setting 中的一個結構性事實。透過這個階梯，我們可以清楚說明為何最終要採用 MDP。

### Tier 1: 單一 TM 設定下，純 MAB 即足以求解

當 TM 為固定常量時，存在一個（或多個等效的）最佳 routing，記為 $a^* \in \arg\max_a \mathbb{E}[r \mid a]$。Multi-armed bandit 的數學樣板**只觀測 reward**（scalar）：每拉一次 arm $a$，環境回傳一個 noisy reward $r$，policy 反覆嘗試後即可收斂到 $\arg\max$ 集合中的某個 element。對應 policy 形式為 $\pi(a)$。

### Tier 1.5: 純 MAB + 同 TM 的 link-obs sequence + memory

仍在固定 TM 設定下：同一個 TM 下，不同 action 會誘導出不同的 link-obs。若允許 policy 把這些 link-obs 累成 sequence、用 RNN 等記憶機制聚合這些觀測，會比純 MAB 強嗎？

**會。** 兩個層面：

1. **Reward variance reduction**：reward 是 noisy scalar、訊息密度低；link-obs 是 vector、結構化、訊息密度高。RNN 聚合多次 obs 可以更精準估計 arm 的 expected reward，達成 reward variance 的縮減（這在統計上對應 *control variate* 技巧——用一個與目標相關、但 variance 較低的旁觀測量去校正主要估計量）。
2. **隱式 TM 識別**：當 TM 偶爾切換時，純 MAB 必須重新學；但 link-obs sequence 自然編碼了當前 TM 的 footprint，RNN 可以從序列中內隱推斷當前所處的 TM，policy 過渡更平滑。

這個設定**仍然不需要 transition kernel 或 $\gamma$ 等 MDP 工具**——context 沒有被顯式當作輸入（TM 不可觀測），而是透過 obs sequence 以隱式形式存在，可視為 MAB 與 contextual bandit 之間的中間形態。

> 為什麼「同 TM 下不同 action 誘導出的 link-obs」不算 contextual bandit 中的 context？這牽涉 contextual bandit 的形式語意，留到 Tier 2 一併釐清。

### Tier 2: 多 TM 設定下，policy 需要顯式 conditioning on link-obs

實際 SDN 部署中，TM 隨時間變動，最佳 routing 因此依當下 TM 而變，policy 必須以某個 proxy 來識別當下處於哪個 TM。Link-obs 雖非 raw TM，卻**間接編碼了當前 TM 的 footprint**——哪些 link 載重高、哪些低——因此自然成為這個 proxy。對應的 policy 形式為 $\pi(a \mid o)$，符合 **contextual bandit** 的數學樣板。

Contextual bandit 的標準假設包括：

1. **i.i.d. contexts**：$o_t \sim \mathcal{D}$ 獨立同分佈
2. **Stationary reward function**：$r_t = R(o_t, a_t) + \varepsilon_t$，$R$ 不隨時間變動
3. **Memoryless reward**：$r_t$ 不依賴歷史 $(o_{<t}, a_{<t}, r_{<t})$

只要這三個假設大致成立，contextual bandit 的演算法（LinUCB、neural CB 等）與 regret 分析都適用。**Tier 2 對應「假設 1–3 大致成立」的工作點。**

> **回應 Tier 1.5 留下的問題**：在 contextual bandit 的形式語意中，**context 標誌的是不同的 TM**——同一個 TM 下不同 action 所誘導出的 link-obs 變異，全部被視為**同一個 context distribution 內部的 stochasticity**，並非 context 切換；只有跨 TM 時 context distribution 才真正改變。Tier 1.5 的 obs 變異全在「同一 TM」這個 context 內，所以雖然用了 obs，仍不算 contextual bandit。

### Tier 3: Queue carry-over 讓 i.i.d. context 失效，升格為弱 MDP

Tier 2 的 i.i.d. context 假設在實際網路中被破壞——低容量瓶頸鏈路上的 queue 殘留讓 $o_{t+1}$ 系統性地依賴於 $a_t$（甚至 $o_t$）。一旦這個依賴存在，$s_{t+1}$ 就承載了**真正的 transition 訊息**：它告訴 agent「上一步動作對 link 狀態的後續影響」。

i.i.d. context 假設一旦失效，contextual bandit 的 regret guarantee 也隨之失效——我們**必須**用更一般的 MDP 框架來處理這個跨步 dependence。這同時意味著問題本身變難了：state 之間的 transition 結構必須被顯式建模、value 必須跨 step 傳遞。MDP 框架對應地提供了處理這個 dependence 的標準工具：用 $s_{t+1}$ 做 TD bootstrap、replay buffer 跨 step 的 reuse、target encoder 做 next-state value estimation。

policy 因此寫成 $\pi(a_t \mid s_t)$，配合 transition kernel $P(s_{t+1} \mid s_t, a_t)$。

但要強調：這個 MDP **不是典型 MDP，而是一個結構特殊的「弱 MDP」**——transition 主要由 action 決定，state 對下一步 state 的影響被局部物理機制（queue carry-over）所限制。沒有 flow lifecycle 動態、沒有 TCP 跨步耦合、沒有「為了應付未來變化提前預留資源」的長程動機（TM 在 epoch 內 stationary）。**這個 MDP 正是後續設計選擇的依據**：$\gamma$ 的設定、replay 的角色、critic 的訓練 dynamics 都從這個結構性導出。

## 3. State 定義與 Action-Determined Reachability 假設

我們直接把 link observation 本身定義為 state：

- **State $s_t$**：第 $t$ 步的完整 per-link 觀測（throughput / delay / loss）
- **Action $a_t$**：對每個 OD pair 從 $K$ 條候選路徑中選一條
- **Transition $s_{t+1} \sim P(\cdot \mid s_t, a_t)$**：新路由安裝、流量穩態後的下一輪 link 觀測
- **Reward $r_t$**：由 $s_{t+1}$ 衍生出的 per-pair 向量（MLU、過載罰項、路徑品質）

此定式的核心結構性假設稱為**近似 action-determined reachability**：

$$
\bigl\| \mathbb{E}[s_{t+1} \mid s_t, a_t]\ -\ \mathbb{E}[s_{t+1} \mid s_t', a_t] \bigr\| \leq \varepsilon \quad \forall\, s_t, s_t' \in \mathcal{S}_{\text{TM}}
$$

其中 $\|\cdot\|$ 是 L2 norm，$\varepsilon$ 為一個小量，$\mathcal{S}_{\text{TM}}$ 是**同一個 TM 下**可達的 link-obs 集合。**直觀解讀**：在同一個 TM 下，給定相同 action，從不同的當前 link-obs 出發，預期下一步觀測差距不大——亦即下一步觀測**主要由 action 決定**，當前 state 的影響被壓在 $\varepsilon$ 之內。跨 TM 之間 obs 的差異不在此假設範圍內，那屬於 Tier 2 的 context shift 問題。

> $\varepsilon$ 並未強行區分兩種來源：(a) 殘留的 state-dependence（即我們認為很小、但實際非零的那部分），(b) 與 $(s_t, a_t)$ 正交的 exogenous noise（測量噪聲、背景流量、timing jitter 等）。我們的假設認為 (a) 小；下文用 queue carry-over 的物理估計給出 (a) 的具體量級，但 (b) 並未在 $\varepsilon$ 中被獨立扣除。

弱化此假設的物理來源明確：**低容量瓶頸鏈路的 queue carry-over**。以 GÉANT 拓撲中 1.55 Mbps link 配 1000-pkt（1488 B）queue 為例，滿 queue 對應的 backlog 可換算成 $\sim 7.68$ 秒的排隊時間（公式與量測詳見 [delay_measurement_issues.md](delay_measurement_issues.md) §1）。這個數字在網路實務中是異常的——一般生產網路的 buffer 都不會大到產生秒級排隊延遲——它源自我們沿用低容量 link 與統一的大 queue size 所導致的實驗設定。相對於 DRL step 的 `MONITOR_PERIOD = 10 s`，這個殘留會明顯延伸到下一步觀測。25 Mbps 以上的 link 同容量 queue 排空時間 $< 0.5$ 秒，影響可忽略。**Reachability 的弱化因此是異質鏈路容量帶來的局部現象**，而非全網均勻擾動。

## 4. Policy-Induced Markov Chain 與近似最優 State 集合 $C$

§3 的近似 action-determined reachability 把 transition kernel 簡化到

$$
P(s' \mid s, a) \approx P(s' \mid a)
$$

——下一步觀測主要由 $a$ 決定，當前 $s$ 的影響被壓在 $\varepsilon$ 之內。進一步地，在同一個 TM 下、給定 $a$ 後，穩態流量於相同候選路徑上會收斂到接近相同的 link 載重，因此存在一個 map

$$
\phi: \mathcal{A} \to \mathcal{S}, \qquad \phi(a) = \text{action } a \text{ 在當前 TM 下穩態後誘導出的 link-obs}
$$

使得 $P(s' \mid a) \approx \mathbb{1}[s' = \phi(a)]$。在不同 routing 於同一 TM 下穩態載重彼此不同的前提下，$\phi$ 是 injective，可逆 map 記為 $\phi^{-1}$（定義域為 $\phi(\mathcal{A})$ 的像集）。

把這兩層近似代入「固定 policy $\pi$ 下 MDP 退化成 Markov chain」的標準推導：

$$
P^{\pi}(s' \mid s) \;=\; \sum_{a} \pi(a \mid s)\, P(s' \mid s, a) \;\approx\; \sum_{a} \pi(a \mid s)\, \mathbb{1}\bigl[\phi(a) = s'\bigr] \;=\; \pi\bigl(\phi^{-1}(s') \,\bigm|\, s\bigr)
$$

**這條誘導 chain 的特殊之處在於**：transition 機率本身就是 policy 在當前 $s$ 下選到「指向 $s'$」的 action 的機率——chain 動態的隨機性完全來自 policy 自己，而非環境的 transition stochasticity；$\pi$ 在此 chain 中扮演的就是一個純粹的 transition probability。

進一步把當前 $s$ 邊際掉，可以得到 $s'$ 的 marginal：

$$
P^{\pi}(s') \;=\; \sum_{s} P^{\pi}(s)\, P^{\pi}(s' \mid s) \;\approx\; \sum_{s} P^{\pi}(s)\, \pi\bigl(\phi^{-1}(s') \mid s\bigr) \;=\; P^{\pi}\bigl(\phi^{-1}(s')\bigr)
$$

也就是說 **state 上的穩態分佈直接等於 marginal policy 在對應 action 上的機率**。又因為每個 $s'$ 都由唯一的 $a = \phi^{-1}(s')$ 誘導、$r$ 又由 $s'$ 直接決定，每個 state 都對應一個固定的 reward 值。**訓練的本質因此可以具體寫成：塑形這條誘導 chain 的穩態分佈，讓質量集中到 reward 高的 $s$ 上**——這嚴格等價於讓 marginal policy 集中到能誘導到那些 $s$ 的 action 上。

### 4.1 訓練目標：從單點收斂擴展為集合 $C$ 的 recurrent dynamics

我們設定的訓練目標**不是讓 chain 收斂到單一最優 state**，而是讓 chain 收斂到一個由多個近似最優 state 組成的集合，記為 **$C$**：

- $C$ 內包含多個能達到同等級 MLU / delay / loss 的 routing configuration
- 訓練收斂後，chain 在 stationary distribution 下的機率質量大部分集中在 $C$；$C$ 中的 state 為 **positive recurrent**——chain 可能因環境噪聲（觀測 jitter、背景流量等）短暫離開 $C$，但 policy 會以高機率把它快速拉回；$C$ 內部則允許在多個近似最優 routing 間自由遊走
- 從外部進入 $C$ 的步數視 reachability 強度而定：嚴格 reachability 下一步即可進入；弱化條件下可能需要兩到三步過渡

> 嚴格而言 $C$ 依當前 TM 而定（記為 $C(\text{TM}_t)$）——不同 TM 對應不同的近似最優 routing 集合。但因為 link-obs 已經間接編碼了當前 TM 的影響，policy 透過 conditioning on $s_t$ 自然處理 TM 切換；後續論述若不特別註明，$C$ 指的是當前 TM 下的近似最優集合。

**這個 set 結構正是 STRIDE（我們的方法）借用擴散模型架構的核心理由**：擴散模型天然支援多峰分佈生成——其反向過程不是收斂到單一 mode，而是把噪聲分佈逐步推到資料流形上的多個 mode，採樣結果在這些 mode 間覆蓋。這恰好對應我們的需求：policy 不應集中在單一 routing 上，而應在 $C$ 的多個近似最優 routing 上維持 mass。把擴散模型的目標分佈從「真實資料分佈」替換成「$C$ 上的近似最優分佈」，即得到 STRIDE 的核心設計思路。

## 5. $\gamma$ 的角色：bandit 等價性與 MDP 增益

$\gamma$（discount factor）的選擇是這個環境建模的另一個關鍵變數。我們從三個角度釐清 $\gamma$ 的功能。

### 5.1 嚴格 reachability 下 $\gamma = 0$ 與 bandit 等價

若 $P(s' \mid s, a) = P(s' \mid a)$，則 $V^*(s)$ 對所有 $s$ 退化為常數，最優 policy 只需 $\arg\max_a \mathbb{E}[r(a)]$，與 $\gamma = 0$ 的 bandit greedy 解一致。**換言之，當 transition 完全由 action 決定時，MDP 的最優解與 bandit 重合**，把問題寫成 MDP 在這個情境下沒有副作用，只是 MDP 的多步規劃能力沒有用武之地；當實際 transition 結構出現（reachability 弱化）時，這個能力才會被用上。

### 5.2 弱化 reachability 下，$\gamma > 0$ 提供多步規劃能力

當部分 state 出發無法一步進入 $C$（低容量 link 的 queue 尚未排空），myopic greedy 會卡在過渡 state。$\gamma > 0$ 透過 bootstrapping 讓 policy 學會兩到三步的 staging：先選一個能加速 queue 排空的 routing，再切到接近 $s^*$ 的最終 routing。**這是 MDP 環境建模相對 bandit 的核心優勢來源**。

### 5.3 即便 reachability 嚴格成立，$\gamma > 0$ 仍提供 variance reduction

我們環境的 reward 量測有噪聲（特別是真實 Mininet 下的 delay / loss 採樣變異）。bootstrapped TD target $r + \gamma \cdot V(s')$ 引入一個獨立、平滑後的 state-value 估計，可以壓低 critic supervision 的 variance。這是與 reachability 強弱無關的純統計 benefit。

## 6. 小結

本 reformulation 確立四個核心 claim：

1. **TM 不可觀測作為硬性 formulation 出發點**——所有後續設計都從這個 partial observation 約束導出，劃清 scope。
2. **層級遞進論證環境建模升級的必要性**——單一 TM 的 MAB / MAB+memory、多 TM 的 contextual bandit、queue carry-over 下的弱 MDP 各對應一個具體的結構性事實。STRIDE 的設計動機落在 Tier 3，但 Tier 1 / 1.5 / 2 同樣是合理的退化建模選項。
3. **訓練目標從「收斂到單一最優 state」擴展為「進入並維持在近似最優 state 集合 $C$」**，policy-induced Markov chain 服務於 RL 控制而非 generative endpoint。
4. **Reachability 弱化的物理來源被收斂到具體量化指標**——在我們目前的 single-flow setting 下，這個指標就是低容量瓶頸鏈路的 queue carry-over 約 7.68 秒。當環境建模推廣到 multi-flow 真實 MDP 場景時，flow lifecycle 動態會引入更豐富的跨步 transition 結構，本 reformulation 的層級遞進論證（特別是 Tier 3 對 transition-aware 工具的需求）對該情境同樣適用、且更貼合典型 MDP 形式。

在這個 formulation 之上，後續章節討論 state 表示、policy 架構、與訓練演算法的具體設計。
