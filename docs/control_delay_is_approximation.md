# 重要性取樣比與控制延遲

STRIDE 的 off-policy A2C 用重要性取樣比 $\rho$ 修正 behavior 與 target policy 的落差。
在這個系統裡，$\rho$ 的**分子與分母條件於不同的 state**。這是控制延遲加上「未將生成態
存進 tuple」的結果。這篇說明它為什麼仍然成立。

> 這份是原始檔。實驗室的 HackMD 上另有一份複本，內容相同，圖片改用 HackMD 自己的
> 上傳。要改就改這裡。

行號都是 [`../algs/stride.py`](../algs/stride.py) 的。

---

## 1. tuple 是延遲修正過的，但少存了一個 state

![Figure 2. Control delay timeline](../paper/figures/control_delay/Figure%202.%20Control%20delay%20timeline.png)

控制延遲的意思是，STRIDE 在週期 $t$ 算出的動作，要到下一個週期才被 controller 部署下去。
所以量到的 $r_{t+1}$ 與 $s_{t+1}$ 不是 $a_t$ 造成的，是 $a_{t-1}$ 造成的。延遲修正過的
tuple 因此是

$$(s_t,\; a_{t-1},\; r_{t+1},\; s_{t+1})$$

**但 $a_{t-1}$ 不是從 $s_t$ 抽出來的。** 它是上一個週期、由 $s_{t-1}$ 生成的，當時存下來的
behavior 機率是

$$\pi_\beta(a_{t-1} \mid s_{t-1})$$

而 tuple 裡**沒有 $s_{t-1}$**。所以要為 $a_{t-1}$ 算現行 policy 的機率時，沒有 $s_{t-1}$
可取，只能拿 tuple 裡有的 $s_t$ 去算。比值於是變成

$$\rho \;=\; \frac{\pi_\theta(a_{t-1} \mid s_t)}{\pi_\beta(a_{t-1} \mid s_{t-1})}$$

分子分母條件於相差一個週期的兩個 state。

### 這是可以避免的，只是沒做

$s_{t-1}$ 並非取不到。多存一個 state，就能讓分子分母都條件於 $s_{t-1}$，比值精確。代價是
每筆 transition 多一份 state 的記憶體，而且該設計未經實測。目前的版本不存，讓它成為近似
—— 下一節是這個近似成立的理由。

## 2. 為什麼近似可接受

### 2a. $s_{t-1}$ 與 $s_t$ 是同一 demand 的兩次下游觀測

這個問題是 spatial POMDP。agent 觀測到的是 link state，那是 traffic demand 的**下游
效應**，真正的 demand（TM）是 latent 的。GRU encoder 把觀測歷史聚合成對 latent demand 的
belief（$h^{\text{pair}}$），policy 實際作用在這個 belief 上，不是 raw link state。

短時間內 demand 視為同一個 epoch、近似常數。$s_{t-1}$ 與 $s_t$ 因此只是**同一個 demand 的
兩次下游觀測**，差別在量測時點。經 GRU 聚合後兩者反映同一個 demand，在 belief 空間相似。
於是

$$\pi_\beta(\cdot\mid s_{t-1}) \approx \pi_\beta(\cdot\mid s_t)$$

$\rho$ 就近似一個分子分母對齊的乾淨比值。誤差量級是
$O(\lVert \text{belief}(s_t) - \text{belief}(s_{t-1}) \rVert)$，在 demand 緩變下小。

此近似與本方法既有的假設一致，並未額外引入條件。以 GRU 近似 POMDP 的 belief、
進而將延遲系統視為 Markov MDP 處理，所依據的正是同一個前提。

### 2b. $\rho$ 是 detach 的截斷權重

$\rho$ 不參與梯度計算，只決定每一項的權重大小，而且截斷於 $[0, 1]$（V-trace
truncation，`:1085`、`:1094`–`:1095`）。2a 的近似誤差因此只改變權重數值，不改變梯度方向
—— 梯度來自 $\nabla \log \pi_\theta(a_{t-1} \mid s_t)$，在生效態上取，與 $\rho$ 無關。

---

> **備註：程式裡的變數命名早一格。** 上面全部用論文的下標。程式碼裡同一組東西寫成
> $s_{t-1}$、$a_{t-2}$、$r_t$、$s_t$，對照如下。看 code 時記得換算。
>
> | 論文 | 程式碼 | 位置 |
> | --- | --- | --- |
> | $s_t$（生效態，存進 tuple） | $s_{t-1}$ | `:1062` 取 `window[t]['state']` |
> | $a_{t-1}$ | $a_{t-2}$ | `:1008` 存 `info_list[0]['action']` |
> | $\pi_\beta$ 條件的生成態 | $s_{t-2}$ | 只存在於 `info_list[0]`，未寫入 replay |
> | $r_{t+1}$ | $r_t$ | `:1092` 算 `delta` |
> | $s_{t+1}$ | $s_t$ | `:1071` 取 `window[t]['next_state']` |
