# Sim fluid queue 模型：從 naive baseline 到經驗校正的三個旋鈕

## 目的

在 simulation 環境 (`environment16.py`) 中加入 **per-link directed queue model**，
使 sim 環境的 delay / loss / queue_pkts 與 real Mininet 環境 (netem-only) 的行為對齊，
讓 RL agent 在 sim 訓練時就能學到真實的擁塞信號。

---

## 設計原理

### Fluid Queue Model（流體排隊模型）

不做封包級模擬，而是用「每 step 的 bit 總量」做流體近似：

```
arrival_bits = utilization (Kbps) × 1000 × step_duration (s)
service_bits = capacity   (Kbps) × 1000 × step_duration (s)

q += arrival_bits
served = min(q, service_bits)
q -= served

if q > max_queue_bits:
    overflow = q - max_queue_bits   ← 溢出的 bits（被丟棄）
    q = max_queue_bits
else:
    overflow = 0
```

- **arrival**：從 `edge['utilization']`（traffic allocation 後的 Kbps）
- **service**：從 `edge['capacity']`（link bandwidth Kbps）
- **queue 上限**：`max_queue_pkts × avg_pkt_bytes × 8` bits

### 為什麼是 transient（每 step 更新一次）

Real env 中，Ryu controller 每 `MONITOR_PERIOD` 秒收集一次 link metrics，
看到的是「這個 step 結束時」的 queue 狀態。
所以 sim 也必須以相同的時間粒度更新 queue，保持 sim-real 對齊。

若用穩態模型（arrival > capacity → queue 直接滿），會失去「queue 逐步累積」的動態過程，
跟 real env 的觀測不一致。

---

## 衍生指標計算

每次 `_update_queues()` 結束後，寫入 graph edge：

| 指標 | 公式 | 說明 |
|------|------|------|
| `delay` (s) | `q_bits / (capacity_kbps × 1000)` | 排隊延遲 = queue 中的 bits ÷ 出口速率 |
| `pkloss` (ratio) | `overflow_bits / arrival_bits` | 溢出比 = 被丟的 bits ÷ 進來的 bits |
| `queue_pkts` (int) | `q_bits / (avg_pkt_bytes × 8)` | queue 中等效封包數 |

### Delay 公式解釋

queue 中有 `q` bits 等待傳出，出口速率 `C` bps，
一個新封包到達後要等 `q/C` 秒才能被服務。
這就是 **排隊延遲 (queuing delay)**。

> 例：q = 11,904,000 bits, C = 1,550,000 bps → delay = 7.68 s
> 對應 queue_pkts = 1000, avg_pkt = 1488 bytes, capacity = 1.55 Mbps

### 網路延遲的四個組成：sim 目前只模擬哪一項

網路上單跳 link 的實際 end-to-end delay 由四項組成，**四項在真實網路中都存在**：

| 延遲類型 | 公式 | 典型量級（1.55 Mbps / 1488 B pkt） | Mininet 實際值 | 本 sim 是否模擬 |
|----------|------|-----------------------------------|----------------|----------------|
| **Queuing delay**（排隊）| `q_bits / C` | 0 – 7.68 s（視 q 而定）| netem 實際排隊 | **✅ 模擬**（本文 fluid queue model）|
| **Transmission delay**（傳送）| `pkt_bytes × 8 / C` | `1488×8/1.55M ≈ 7.7 ms` / pkt | HTB rate-shape 真的會 serialize（實際有）| ❌ **未模擬**，未來可加 |
| **Propagation delay**（傳播）| `distance / signal_speed` | 實機依拓撲；Mininet veth ≈ 0 | veth pair 跨 kernel namespace，propagation ≈ μs，實務上視為 0 | ❌ 未模擬；`bw_r.txt` 第三欄保留位但未讀取（待辦）|
| **Processing delay**（處理）| 查 table / 檢 header 固定開銷 | μs 級（OVS datapath fast-path）| Mininet 內核轉發 ≈ μs | ❌ 未模擬（量級太小，可永久忽略）|

**為什麼目前只管 queuing**：在壅塞 link 上 queuing 是絕對主導項 — TM03 壅塞時 queuing ≈ 7.68 s，其他三項加起來不到 10 ms，差了近 1000×。tc backlog 當 ground truth 量的也只是 queuing 那一段（`backlog_bytes × 8 / C`），因此 sim vs real 對比時只需要對齊 queuing 就夠了，這也是本文 calibration 的範圍。

**各項未模擬的後果**：

- **Transmission delay**：非壅塞 link 上本來會有一個 `pkt/C` 的 floor delay — 在本 sim 裡這些 link 的 `delay = q/C ≈ 0`，比 real 少 ~7.7 ms/pkt。壅塞 link 上 queuing 就已經把這一項淹沒（7680 ms vs 7.7 ms 差 1000×），不影響校正結果；**若未來要跑更細的路徑級 delay 預測，這一項需要補進 sim** — 簡單加法：`delay_out = q/C + pkt/C = (q + pkt) / C`。
- **Propagation delay**：Mininet 本身接近 0 所以 sim=0 沒有差距，但若以後切到真實長距 link（或用 `tc netem delay X ms` 給每條 link 加固定 delay），需要從 `bw_r.txt` 第三欄讀入 per-link 固定值，加在 `delay = q/C + pkt/C + prop_delay`。
- **Processing delay**：OVS datapath fast-path 是 μs 級，即使把所有 link 加起來跑一條 10-hop 路徑也不到 1 ms，和 queuing 量級差太多，**長期都不用模擬**。

### Loss 公式解釋

當 queue 已滿 (`q > max_queue_bits`)，超出的部分就是 tail-drop loss。
loss ratio = 被丟的量 / 總到達量，跟 real env netem 的 tail-drop 行為一致。

---

## 參數設定：從 naive 抄寫到經驗校正

### Naive baseline — 直接套用 real env 帳面參數

最原始設計時，我們把 real env 上看得到的數字直接抄過來、不做任何 wire-overhead 或 kernel-counter 層面的修正。這個 naive 版本只是為了驗證 fluid queue 整體框架是否可跑、可收斂，並沒有宣稱對齊 real env 的量化行為；後續的所有校正都是在這個起點上逐項修正：

| 參數 | **Naive（直接抄 real env）** | 抄寫依據 |
|------|--------------------------|---------|
| `queue_step_duration` | 10s | 直接用 `MONITOR_PERIOD` |
| `queue_max_pkts` | 1000 | 直接用 netem `limit=1000` |
| `queue_avg_pkt_bytes` | 1460 | iperf3 `-b` 設定的 UDP payload size |
| `queue_overhead_factor` | 1.0 | 乘在 demand 上的係數（`arrival_bps = demand × overhead`）；naive 預設 1.0 = 假設應用層 demand 就等於 netem 看到的 arrival bits，沒考慮 L3 header 等額外 bits — 詳細見 §(b) |
| pre-fill steps | 0 | 未考慮 traffic warmup 比 DRL step 0 早啟動 |

### Naive baseline vs 校正後 vs Real — 圖示對比

下圖把三條曲線疊在同一張圖上（link 13→14, TM03, 1.55 Mbps），直觀呈現 naive 版本跟 real 的三種誤差是如何顯現出來的：

![naive_vs_cal_vs_real_13_14_tm03](https://hackmd.io/_uploads/BJco8B62bg.png)


> **座標軸約定**：圖上 step N 點的 y 值定義為「**DRL 在 step N 做決策那一瞬間觀測到的 queue 狀態**」— 也就是 step N 自己的 `_update_queues()` **還沒跑**之前、從前一個 step（或 step 0 之前的 pre-fill 階段）遺留下來的 q。

> 右軸 queue_pkts 是左軸 delay 經由 `pkt=1488 B, C=1.55 Mbps` 線性換算，兩軸是同一條曲線的不同單位。

> **術語澄清 — queue-fill 階段 vs iperf3 pacing ramp-up**：
> - **Queue-fill 階段**：q 還在線性累積、未撞到 `limit` 的斜坡區段，本文中所有「爬升」「ramp-up」若無特別說明一律指這個。壅塞 link 上持續 ~13 step。
> - **iperf3 pacing ramp-up**：UDP client 首 ~1 秒內 bitrate 還沒衝到 `-b` 設定值那一瞬，只發生在 traffic 啟動時。
> 具體影響：real 從 t=40 iperf3 啟動到 t=60 DRL step 0 第一次讀 backlog，中間有 20 s = 2 個 step 的累積時間，純線性理論預測 step 0 觀測值 ≈ 2 × 560 ms ≈ 1120 ms；但實測 892 ms，少了約 230 ms。這 230 ms 就是 iperf3 首秒 pacing 還沒拉滿造成的「有效累積時間」比 2 step 少 ~0.4 step。這也是後面校正選 pre-fill=1（而非 2）的原因。

- **校正後（藍）** 與 **real tc（黑）** 幾乎完全重疊：step 0 初始值差 332 ms（real 892 vs cal 560，因為 pre-fill=1 step 對應的 560 ms 略小於 real 的 ~1.6 effective step），slope、saturation plateau 都對齊到毫秒等級。Calibrated 在 step 13 進入 saturation、real 在 step 12，差 1 step 就是 initial 332 ms gap 延後 saturation 到達所致。
- **Naive（紅虛線）** 錯了三件事（各自對應一個參數，在圖上表現為三個獨立的 geometric error）：
  - **起點錯**：step 0 從 0 ms 出發 — 沒有 pre-fill 且假設 DRL step 0 就是 traffic 啟動時刻，real 實際上已累積 892 ms 排隊
  - **斜率錯**：slope 只有 real 的 42% — 每步 delay 增量 232 ms vs real 560 ms
  - **上限錯**：saturation plateau 7535 ms vs real 7675 ms，差 140 ms — `avg_pkt_bytes` 用 UDP payload 1460 而不是 L3 skb 1488 造成的。另外還有一個反向誤差來源：netem `limit=1000` 是封包數上限，控制封包（LLDP/ARP/OpenFlow）也會佔 slot 讓實際 data 能累的 byte 上限略低 — fluid model 不處理這一項，直接假設所有 slot 都是 data（見下方 §「一個簡化假設」）。

> 繪圖腳本 `stats/plot_naive_vs_calibrated.py` 已於 2026-08-18 移除（輸入 session 早已不存在，無法重跑）。以下圖表為當時產出。 
sim 曲線：實際跑 `test_sim_only --ospf` 。
Real 曲線 `load_real_tc()` 直接讀 `results/ospf/test/20260414_120638/real/03/probe_vs_tc.csv` 的 `tc_delay_ms` 欄，這個 CSV 就是 §loss/delay 文件中交叉驗證實驗用的同一份資料。

### 控制公式：每個參數各自進入曲線的哪個位置

Fluid queue 的三個校正旋鈕（`avg_pkt_bytes`、`overhead_factor`、`pre_fill`）彼此正交，各自只影響曲線的**一個**幾何特徵。把 `_update_queues()` 展開成封閉式，參數位置就一目了然：

```text
(1) 每 step 淨增量      Δq_bits    = (demand_kbps × overhead_factor − capacity_kbps) × 1000 × step_duration
(2) queue 飽和上限      Q_max_bits = queue_max_pkts × avg_pkt_bytes × 8
(3) 單次 update         q_{k+1}    = clip(q_k + Δq_bits,  0,  Q_max_bits)    # clip 同時處理 underflow 與 tail-drop
(4) 初始條件            q_0 = 0   (traffic 啟動時 queue 為空)
(5) step N 觀測值（決策前） q(N) = q_{pre_fill + N}                           # 從 q_0 連跑 (pre_fill + N) 次 update
(6) 換算成延遲           delay_ms(N) = q(N) / (capacity_kbps × 1000) × 1000
```

每個參數只出現在其中一條線上 → 直接對應圖上的一個特徵：

| 參數 | 出現在公式 | 影響的圖形特徵（主）| 次要/間接效果 | Naive→校正的移動方向 |
|------|------------|------------------|--------------|-------------------|
| `avg_pkt_bytes` | (2) | **saturation plateau 高度（左軸 delay_ms）** — 垂直拉高/壓低水平上限線 | 也出現在右軸 `pkts` 換算（`q/(pkt×8)`），所以 **fill rate 以 pkts/step 表示時也會改** — 左軸 ms/step 不變；間接透過 `Q_max_bits` 影響「多久撞到上限」 | 1460→1488：plateau **抬高** 7535→7680 ms；fill rate 從 30.8→30.2 pkts/step（若只動 pkt） |
| `overhead_factor` | (1) | **queue-fill 段斜率（ramp-up slope）** — 爬升段傾斜角 | 透過 Δq_bits 決定 steps-to-saturation | 1.0→1.032：slope **變陡** 232→560.5 ms/step |
| `pre_fill` | (5) | **step 0 起始高度** — 整條曲線沿 x 軸左移 pre_fill 個 step | 間接延後/提前撞上限 | 0→1：step 0 從 0 ms **抬到** 560 ms |
| `queue_max_pkts` | (2) | saturation plateau 高度（與 `avg_pkt_bytes` 相乘）| — | 維持 1000，無變動 |
| `step_duration` | (1) | 單位時間刻度（ms/step 斜率隨之等比例變化）| — | 維持 10s，無變動 |

直觀一句話：**`overhead_factor` 改 queue-fill 斜率、`avg_pkt_bytes` 改上限（並微調 pkts-軸斜率）、`pre_fill` 改起點**。

### 數值分解：Naive 三項誤差各來自哪個參數

套用上面的公式，naive 三項誤差各自拆解到一個參數：

| 指標 | Naive 預測 (含計算過程) | Real (tc) | 誤差 | 主要肇因 |
|------|------------------------|-----------|-----|---------|
| saturation delay | `1000 pkts × 1460 B × 8 / 1,550,000 bps = 7,535 ms` | 7,675 ms | **−140 ms (−1.8%)** | `avg_pkt_bytes` 用 UDP payload 1460 而非 L3 skb 1488 |
| slope (ms/step) | excess bps = (1586 − 1550)×1000 = 36,000 bps; per step (10s) = 360,000 bits; `360,000 / 1,550,000 × 1000 = 232 ms` | 560 ms | **−328 ms (−58.5%)** | `overhead_factor=1.0` 完全沒考慮 arrival 在 L3 skb 層看到的額外 bits |
| fill rate (pkts/step) | `360,000 bits / (1460 × 8) = 30.8 pkts` | 73 pkts | **−42 pkts (−57.8%)** | 同上（跟 slope 同一個錯誤） |
| steps to saturation | `1000 pkts / 30.8 pkts/step = 32.4 steps` | ~13 steps | **2.5× 過慢，30-step 視窗內永遠沒飽和** | slope 被低估 |
| step 0 delay（observation） | 0 ms（pre-fill=0，step 0 開始前沒有任何累積） | 892 ms | **−892 ms** | 沒把 traffic 在 DRL step 0 之前 ~1.6 effective step 的 warmup 算進來 |

Naive 版本下：queue 在 30-step 實驗視窗裡 **從頭到尾都在 ramp-up、完全沒進入 saturation/tail-drop 區段**、step 0 狀態完全是空的、slope 也只有 real 的 42% — 跟 real 環境的時序、slope、初始狀態三個層面同時偏離，RL agent 在 sim 上看到的壅塞動態跟實機會是兩件事。

### 三個校正及其物理意義

從 naive 到現行參數值的變更逐項對應一個具體的 modeling error：

**(a) `avg_pkt_bytes`: 1460 → 1488** — 修正 tc 計量層級

tc htb/netem 的計數器是在 **L3 skb** 層面累計（packet 進入 qdisc 時 skb 的 size 欄位），skb 含 IP header(20) + UDP header(8) + payload(1460) = **1488 B**，不含 Ethernet header(14B)。用 1460 等於只把 payload 算進 queue，低估了 queue 的實際 byte 佔用。

```
max_queue_bits = 1000 × 1488 × 8 = 11,904,000 bits
saturation_delay = 11,904,000 / 1,550,000 = 7680 ms
→ 誤差 7680 − 7675 = +5 ms ≈ +0.07%
```

> 對比：若用 L2 frame size 1502B → 7752 ms（誤差 +1.0%）— 證實 tc 沒把 Ethernet header 算進去。

**(b) `overhead_factor`: 1.0 → 1.032** — 修正在 L3 skb 層看到的額外 arrival bits

iperf3 的 `-b` 指定的是 UDP **payload** 發送速率（應用層 bps）。但 tc / netem 的計數是在 **L3 skb 層** — 封包進 qdisc 時 skb 的 `len` 欄位 — 所以會看到：

- **IP + UDP header（28 B / 1460 B ≈ 1.92%）** — 每個 payload 封包在 L3 都會多出 28 B header，這是 1.032 中最主要的貢獻
- **LLDP / ARP / OpenFlow echo 等控制流量** — 這些封包本身 bps 很低（LLDP 一條 ~150 B × 每 8 s 一次，bps 換算只有幾十 bps），對 bits-arrival 幾乎可以忽略
- **tc 的 skb byte accounting 對某些 skb 會做 round-up（`qdisc_pkt_len()` 可能含 tail padding 或 SG 對齊）**
- **iperf3 UDP pacing 的誤差，實際送出略高於 `-b` 設定值**

> 注意：Ethernet header (14 B)、preamble (8 B)、IFG (12 B) 是 **L2/wire 層**的額外 bytes，tc htb/netem 在 L3 skb 上計量**根本看不到這些**，所以不列入 overhead_factor 的 arrival 計算。saturation delay 那一側是另一個問題 — 我們透過 `avg_pkt_bytes = 1488`（L3 skb size）就已經正確反映 tc 的計量單位，跟 L2/wire overhead 無關。

純粹由「IP+UDP header = 28 B / 1460 B」推出的理論 overhead 是 1.0192，實測校正值 1.032 比理論高 1.26%，這 1.26% 落在上面後三項（控制流量 + tc rounding + UDP pacing 誤差）的範圍內，個別佔比小、很難精確拆解 — 我們直接以 real 觀測到的 fill rate **反推整體係數**，用 TM03 / 13→14 / OSPF 這一條資料的 73 pkts/step 做 anchor：

```
real fill rate = 73 pkts/step (link 13→14, TM03, OSPF, session 20260413_130143)
excess_bits/step = 73 × 1488 × 8 = 869,184
arrival_bits = excess + service = 869,184 + 1,550,000 × 10 = 16,369,184
demand_bits  = 1,586,000 × 10 = 15,860,000
overhead_factor = 16,369,184 / 15,860,000 ≈ 1.032
```

> **這條 anchor 是單一 link × 單一 TM × 單一 routing 的校正點**。拿同一條 link 的 fill rate 回推 overhead，然後再在**同一條 link 上**驗證 slope 誤差 < 0.1% — 這純粹是**自我校對**，本身不具說服力。後面「8 TM 全量驗證」會用 TM 00/01/02（**獨立的**壅塞 TM）跑同一組 overhead_factor，r² ≥ 0.997 — 那才是排除 overfit 的依據。讀到這裡請把「slope 誤差 +0.1%」當成 sanity check，不是 generalization 證據。

校正後 slope（在 TM03 anchor 上）：
```
Δq_bits = (1586 × 1.032 − 1550) × 1000 × 10 = 868,720 bits/step
slope   = 868,720 / 1,550,000 = 560.5 ms/step  (real 560 → +0.1%, on anchor)
```

### 一個簡化假設：控制封包佔 queue slot 的問題

netem 的 `limit=1000` 是**封包數**上限，不管封包大小 — 一個 ~150 B 的 LLDP frame 和一個 1500 B 的 iperf3 data 封包各佔一個 slot。但我們的 fluid queue model 只有 bits 維度：以 `max_queue_bits = 1000 × 1488 × 8` 當 queue 上限，等於假設**所有 queue 內的封包都是 ~1488 B 的 data**。

實際上壅塞 link 上的 queue 絕大多數是 iperf3 data（~99% 以上的 pkts），控制封包雖然各佔一個 slot 但總數極少，對 saturation delay 的影響可以忽略。我們 **沒有** 另外做 queue slot 會計，選擇用 overhead_factor 把控制流量的 bits 一併塞進 arrival — 這是一個**刻意的簡化**：只要我們關心的指標是 saturation 大略對齊、slope 大略正確，就不需要把 netem packet-limit 的精確語意搬進 fluid model。若未來要模擬非 UDP-heavy 的場景（例如大量小封包的 control-plane traffic），這個假設需要重新檢視。

**(c) pre-fill: 0 → 1 step** — 修正 DRL step 0 ≠ traffic 啟動時刻

`test_single_tm` 的真實時序是：t=40 iperf3 啟動 → t=60 DRL step 0；即 step 0 記錄下來之前 queue 已經累積了 ~20 秒的 SP traffic。但 iperf3 首秒 pacing ramp-up + flow-install 邊界時間讓有效累積約為 **1.6 step**（實測 step 0 backlog = 118 pkts 而非理論 146 pkts），取 **1 step pre-fill** 最接近。詳細時序見 §「Sim vs Real 時間線對齊」。

### 三個校正都獨立可觀察、互不干擾

| 指標 | 只取決於 |
|---|---|
| saturation delay | `avg_pkt_bytes × queue_max_pkts` |
| slope / fill rate | `overhead_factor`（在 demand > capacity 的區段）|
| step 0 初始值 | pre-fill steps |

因此三個錯誤可以從 real 曲線上直接讀出並獨立校正，不會相互掩蓋。

### 校正前後數值彙整

| 指標 | Naive | **校正後** | Real (tc) |
|------|-------|----------|-----------|
| slope (ms/step) | 232 | **560.5** | 560 |
| saturation delay (ms) | 7535 | **7680** | 7675 |
| fill rate (pkts/step) | 30.8 | **73** | 73 |
| steps to saturation | 32 (超出 30-step 視窗) | **13** | 12 |
| step 0 delay (ms) — state at start of step 0 | 0 | **560** | 892 |
| slope 誤差（TM03 anchor）| −58.5% | **+0.1%** | — |
| saturation 誤差 | −1.8% | **+0.07%** | — |

> Step 0 校正後 560 ms 跟 real 892 ms 仍有 332 ms gap，這是因為 pre-fill=1 step 對應的 real warmup 其實是 ~1.6 step。取整為 1 是經驗折衷（取 2 會讓 step 0 過衝到 1120 ms，比 real 多 230 ms）。之後 slope 對齊後這個 initial offset 會持續整個 ramp-up，體現在 saturation 到達時機上 — real step 12 飽和、cal step 13 飽和，差 1 step 就是這 332 ms 沒有被 pre-fill 吃掉的延後。

### 現行參數（2026-04-14 校正後）

| 參數 | 值 | 來源 |
|------|----|------|
| `queue_step_duration` | 10s | `utils/setting.py` `MONITOR_PERIOD` |
| `queue_max_pkts` | 1000 | netem `limit=1000` |
| `queue_avg_pkt_bytes` | 1488 | L3 skb = IP(20) + UDP(8) + payload(1460) |
| `queue_overhead_factor` | 1.032 | 經驗校正，對齊 real 73 pkts/step fill rate |
| pre-fill | 1 step | 對齊 iperf3 warmup 有效累積 ~1.6 step |

這些參數可以透過 `kpath_cfg` dict 覆寫（2026-08 之前叫 `maskgit_cfg`）。若要重現上表的 naive baseline 跑一次對照實驗，將四個 cfg key 設為 `{queue_avg_pkt_bytes: 1460, queue_overhead_factor: 1.0, pre-fill=0}` 即可。

---

## Eval 模式

### Steady-State Snapshot（Training Eval）

`compute_network_metrics_nx(skip_queue=False)` 用於 training 離線 eval，流程：

1. **Save** training 的 `queue_bits` 狀態
2. **Reset** queue 歸零
3. 寫入 load（drl_paths + TM）到 graph edges
4. **跑 30 步** `_update_queues()` → 達到 steady state
5. 讀取 delay / loss / queue_pkts
6. **Restore** training 的 queue_bits（不干擾 training 狀態）

這樣 eval 拿到的是「此 routing 在 steady state 下的 metrics」，
而 training loop 中 queue 的累積不受影響。

### Transient Mode（test_single_tm / test_sim_only）

`run_eval(transient=True)` / `run_sim_eval()` 的流程：

1. `write_load_to_graph()` — 寫 directed utilization 到 graph edges（一次）
2. `_update_queues()` — 推進 queue **1 步**（一次）
3. `compute_network_metrics_nx(skip_queue=True)` × 2 — 讀 undirected + directed metrics

Queue 狀態**跨 step 保留**，不 reset、不 restore。
這讓 sim 的 queue 逐步累積，對齊 real env 的 transient 行為。

| Mode | 用途 | Queue 行為 | 呼叫者 |
|------|------|-----------|--------|
| Steady-state | Training eval（離線） | reset + 30 步 + restore | 已隨 wandb 一併移除 |
| Transient | test_single_tm / test_sim_only | 每步推 1 步，跨 step 保留 | `run_eval(transient=True)`, `run_sim_eval` |

#### test_single_tm 額外校正

`testing_ma` MASKGIT 分支每步呼叫 `apply_netinfo_directed_to_graph()`，
把 real net_info_directed.csv 的 utilization / delay / pkloss / **queue_pkts** 回填到 sim graph。
這確保 sim state 在每步開頭跟 real env 同步，queue_bits 也從 real 校正。

---

## 程式碼整合點

### 1. Queue 狀態初始化 — `generate_environment()`

```python
self.queue_bits = {}
for i in self.graph:
    for j in self.graph[i]:
        self.queue_bits[(i, j)] = 0.0
```

每條 directed link 一個 queue，初始為空。

### 2. Queue 重置 — `reset_queues()`

在 `clean_and_generate_tm()` 裡呼叫，TM 切換時（新 episode）清空所有 queue。

### 3. Queue 更新 — `_update_queues()`

在兩個地方呼叫：
- `rollout_full()` — MaskGIT 的 traffic allocation 後
- `allocate_by_ma()` — DRL routing allocation 後

### 4. 讀取 queue 指標 — `reset_and_get_state_by_NX()`（**僅 undirected 路徑使用**）

此函式以 `self.undirected_edges` 迭代、對每條物理 link 把兩個方向的 `delay` / `pkloss` 以 `max` 合併、`utilization` 以兩方向 clamp 後加總 — 產出的 state 維度為 `num_undirected × 3`。從構造上就只能給 undirected agent 用。

```python
# 迭代 undirected_edges，對兩方向 max
for idx, (u, v) in enumerate(self.undirected_edges):
    delay_uv = self.graph[u][v][0].get('delay', 0.0)
    delay_vu = self.graph[v][u][0].get('delay', 0.0)
    cur_delay  = max(delay_uv, delay_vu) + 1e-6
    cur_pkloss = max(loss_uv, loss_vu)
```

**使用者**（確認範圍 — `grep reset_and_get_state_by_NX `）：
- `loader/train_loader.py:199`（`train()` 主迴圈）
- `loader/train_loader.py:324`（testing / eval 入口）

**`by_NX` 只會被「old MA」與「MaskGIT-undirected」共享呼叫**。MaskGIT-directed 分支若要啟用 directed state，不會走這個函式 — 需要另寫一個以 `self.graph.edges()`（directed MultiDiGraph 邊）迭代、**不做 max 合併**的 state 函式。目前倉庫內沒有這樣的 directed state builder（`grep -rn "directed_edges\|directed.*get_state"` 回空），表示 directed state 路徑尚未實作；只有 **metrics 計算**側（`compute_network_metrics_nx(directed=True)`）有 directed 支援，那是用來產生 eval CSV、不餵給 agent。

> 用 `max` 而非 `mean` 是為了保留「任一方向壅塞」訊號，見 [`directed_vs_undirected_metrics.md`](directed_vs_undirected_metrics.md) §2；但 max 仍屬無向指標、無法區分非對稱壅塞的方向性，因此 directed agent 一旦接上，務必另走直接讀 `graph[u][v][0]` 的路徑。

---

## 與 Real Env 對齊

| 維度 | Real (Mininet + netem) | Sim (fluid queue) |
|------|------------------------|-------------------|
| Queue 機制 | netem `limit=1000` pkts, tail-drop | `max_queue_bits` 溢出截斷 |
| Delay 來源 | LLDP 測量 (系統性低估, 詳見下方) / tc backlog (ground truth) | `q / C` 計算 |
| Loss 來源 | netem tail-drop (tc stats per-step delta) | overflow / arrival |
| 觀測週期 | `MONITOR_PERIOD` (10s) | 同 |
| Buffer 大小 | 1000 pkts (netem limit) | 1000 pkts × 1488B |
| 封包大小 | L3 skb = 1488B (tc htb 計量) | 1488B (參數) |
| Fill time (1.55Mbps) | ~13 steps (real 觀測) | ~13 steps (模型計算) |
| Saturation delay (1.55Mbps) | 7675 ms (real tc) | 7680 ms (sim) |

### Delay 差異說明：LLDP 系統性低估與 tc/UDP 驗證

Sim delay 是理論排隊延遲 `q/C`（queue 全滿時 ~7.4s for 1.55Mbps link）。

**LLDP 不適合作為 delay calibration 基準。** 2026-04-13 的 UDP probe + tc backlog 交叉驗證
（`test_single_tm_udp.py`, session `20260413_130143`）量化了 LLDP 的系統性偏差：

| 測量方式 | Link 13→14 steady-state mean | 佔 tc 比例 | 偏差來源 |
|----------|------------------------------|-----------|----------|
| **tc backlog** | 6471 ms | 100%（基準） | kernel queue 瞬時值 |
| **UDP probe fwd** | 5879 ms | 90.8% | 封包實際排隊體驗 |
| **LLDP fwd** (directed) | 2494 ms | **38.5%** | 每 5s 取樣 + 反映歷史 queue |
| **LLDP undirected** (fwd+rev)/2 | 1248 ms | **19.3%** | 再被空閒反方向平均 |
| **Sim q/C** (corrected) | 6304 ms (steady mean) | **97.4%** | queue sync from real + 1 step advance |

LLDP 低估的兩個系統性原因：
1. **取樣偏差 (~2.6×)**：LLDP 每 ~5s 發一次 frame，frame 進入 queue 後排在已有封包後面。
   測到的 delay 反映 frame 進 queue 瞬間的 depth，不是當前瞬時值。
2. **方向平均 (2×)**：undirected 公式 `(fwd+rev)/2`，壅塞方向被空閒反方向稀釋。

**結論：sim 的 `q/C` 比 LLDP 更接近 tc ground truth。**
以 queue fill rate 和 loss onset 時間做 sim-real 對齊，不以 LLDP delay 值做 calibration。

詳見 `docs/delay_measurement_issues.md`。

---

## Sim vs Real 時間線對齊

### Real Env 時間線（test_single_tm）

```
t=0    build_topo + spawn_controller (Ryu 開始 monitor_period=10s cycle)
t=30   spawn_drl + start_single_traffic
t=40   iperf3 clients start → ★ 流量開始 ★ (SP routing from init_paths)
       queue 開始累積 (SP, 第 1 步)
t=50   Ryu met_1: 寫 net_info (SP traffic t=40→50)
       queue 繼續累積 (SP, 第 2 步)
t=60   DRL step 0: 讀 net_info → ★ 第一次推論 ★
       DRL 寫 drl_paths.json
       Ryu met_2: 抓到 drl_paths → 安裝 DRL routing
       但 met_2 的 metrics 仍是 t=50→60 的 SP 結果
t=70   Ryu met_3: ★ 第一次反映 DRL 決策的 metrics ★
       DRL step 1: 讀 met_3
t=80   Ryu met_4: ★ 穩定反映 DRL 決策 ★ (offset=2 from step 0)
```

**關鍵**：DRL step 0 觀測到的 state，已有 **~2 步 SP traffic 的 queue 累積**。

### 經驗校正：1 步 pre-fill (not 2)

名義上 t=40→60 有 2 個 monitor cycle 的 SP traffic，但 2026-04-13 實測顯示
DRL step 0 時 real tc backlog 只有 **118 pkts**（對應 ~1.6 effective steps，
不是預期的 146 pkts）。原因：

- iperf3 UDP 啟動有 ~1 秒 ramp-up（第一秒 bitrate 未達目標）
- Ryu flow-install 在 met_2 (t=60) 安裝 DRL routing 前，SP flow 其實也還在
  install 中的邊界時間（控制器第一次 install 約 t=40~45）
- 因此實際 SP traffic 有效時間 ≈ 16 秒而非 20 秒

結論：**sim 只需 1 步 pre-fill** 就能對齊實際 queue 累積。

### Corrected Sim 時間線（test_single_tm 的 sim eval）

每個 DRL step N:
1. `kpath_reset` → `rollout_full` → `_update_queues` [advance 1, 即 1 步 pre-fill]
2. `apply_netinfo_directed_to_graph` → **覆寫 queue_bits 為 real 值** (advance 被丟棄)
3. `run_eval(transient=True)` → `write_load_to_graph` + `_update_queues` [advance 1 (effective)]
4. 寫入 sim CSV

有效 queue advance: **每步 1 次**（前面的被 real state 覆寫）。
Corrected sim 的 delay 反映：從 real queue 出發 + 1 步新 routing 的 queue 變化。

### Pure Sim 時間線（test_sim_only --ospf）

```
clean_and_generate_tm → queue = 0
write_load_to_graph (SP routing) → set utilization
_update_queues × 1   → pre-fill (1 步，對應 real env ~1.6 step 有效累積)

for step in range(N):
    _update_queues × 1  → 1 步 advance
    compute metrics (skip_queue=True)

Total: 1 (pre-fill) + N (steps) queue advances
```

### ⚠️ test_sim_only 原始 DRL loop 的 double-advance 問題

原始 DRL loop 中每個 step:
- `kpath_reset` → `rollout_full` → `_update_queues` [+1]
- `run_sim_eval` → `write_load` + `_update_queues` [+1]
- = **2 advances/step**

30 步共 3 + 29×2 = **61 advances**（應為 32）。

**影響**：queue 以 ~2× 速度填滿，steady-state delay 偏高。
**修正**：`--ospf` 模式使用獨立 loop，確保 1 advance/step。
DRL 模式暫保留 double-advance（部分模擬 control delay 下新舊 routing 交替）。

### Pure Sim 實驗結果 — 校正後 (2026-04-14)

`test_sim_only --ospf` 對 session `20260414_120638` 的 8 個 TM 跑 30 步 pure sim
（`overhead_factor=1.032`, `avg_pkt_bytes=1488`, `queue_max_pkts=1000`, 1 步 pre-fill）。

#### TM03 link 13→14 (1.55 Mbps) 逐步對比

| Step | tc (real) | sim pure | diff | 備註 |
|------|-----------|----------|------|------|
| 0 | 907.5 (118p) | 1054.1 (136p) | +147 ms | iperf3 ramp-up 導致 real 略低 |
| 2 | 2022.6 (263p) | 2168.6 (280p) | +146 ms | ramp-up 影響漸消 |
| 4 | 3161.0 (412p) | 3283.1 (424p) | +122 ms | slope 對齊，初始偏移逐漸收斂 |
| 6 | 4299.2 (560p) | 4397.6 (568p) | +98 ms | 穩定 ~560 ms/step slope |
| 10 | 6544.8 (852p) | 6626.6 (856p) | +82 ms | 累積誤差穩定 |
| 12+ | 7675.3 (999p sat.) | 7680.0 (1000p sat.) | **+5 ms** | ★ saturation 近乎完美 |

**校正前 vs 校正後**：

| 指標 | 校正前 (04-13) | **校正後 (04-14)** |
|------|---------------|-------------------|
| slope | 527 ms/step (低 6%) | **560.5 ms/step (誤差 0.1%)** |
| saturation delay | 7378 ms (低 3.9%) | **7680 ms (誤差 0.07%)** |
| step 12+ diff | -297 ms | **+5 ms** |
| step 4 diff | +0.7 ms | +122 ms (初始偏移，非 slope 問題) |

#### 8 TM 全量驗證 — summary (step ≥ 3, all links)

校正用 TM03 數據，但 TM 00/01/02 也有 SP MLU > 1.0（同樣壅塞 link 13→14），
作為**獨立驗證**確保 queue model 不是 overfit：

| TM | SP MLU | sim r² vs tc | sim MAE (ms) | sim bias (ms) | 壅塞邊 |
|----|--------|-------------|-------------|--------------|--------|
| 00 | 1.1818 | 1.000 | 1.000 | -0.2 | 13→14 (1.55M) |
| 01 | 1.1606 | 1.000 | 0.7 | -0.2 | 13→14 + 11→10 (25M) |
| 02 | 1.1253 | 1.000 | 0.7 | -0.2 | 13→14 (1.55M) |
| 03 | 1.0266 | 0.997 | 1.000 | +2.5 | 13→14 (1.55M) |
| 10 | — | 0.985 | 2.8 | +2.5 | 無壅塞 |
| 12 | — | 0.141 | 0.5 | +0.0 | 無壅塞 |
| 14 | — | 0.254 | 0.5 | — | 無壅塞 |
| 21 | — | 0.330 | 0.6 | 0.2 | 無壅塞 |

**壅塞 TM (00-03)**：r² ≥ 0.997, bias < 3 ms — 校正在獨立 TM 上完全成立。
**非壅塞 TM (10-21)**：delay 本身極小 (< 1 ms)，r² 低是因為 signal-to-noise 比差，
MAE 均 < 3 ms，不影響實用性。

**結論**：fluid queue model 校正後，slope 和 saturation 同時精確對齊 real env。
3 個獨立 TM 的壅塞邊驗證 r² ≥ 0.997，確認 overhead_factor=1.032 + avg_pkt_bytes=1488
不是 overfit TM03 的結果。

繪圖：`stats/plot_presentation.py`（已於 2026-08-18 移除，所需 session `results/ospf/test/20260414_120638` 亦已不存在）
輸出：`artifacts/plots_presentation/`（已隨腳本移除）

---

## 數值範例

以 link 13→14 為例（capacity = 1.55 Mbps，demand ≈ 1.586 Mbps，overhead_factor = 1.032）：

```text
effective_arrival = 1586 × 1.032 = 1636.8 kbps
max_queue_bits = 1000 × 1488 × 8 = 11,904,000 bits
excess_bits_per_step = (1636.8 - 1550) × 1000 × 10 = 868,000 bits/step
excess_pkts_per_step = 868,000 / (1488 × 8) ≈ 73 pkts/step

Step 1:  q = 868,000 → delay = 0.56s, loss = 0%, queue_pkts = 73
Step 2:  q = 1,736,000 → delay = 1.12s, queue_pkts = 146
...
Step 13: q = 11,284,000 → delay = 7.28s, queue_pkts = 948 (尚未滿)
Step 14: q = 12,152,000 > 11,904,000 → 開始 loss!
         q = 11,904,000 (capped), queue_pkts = 1000
Step 15+: steady state, loss ≈ 5.3%, queue_pkts = 1000
```

> 對應 real env：queue 從 ~118 pkts 開始，~73 pkts/step 累積，step 12 到 999 開始 loss (~5.3%)
> Saturation delay: sim 7680 ms vs real 7675 ms（誤差 0.07%）

---

## 未來優化

1. **Propagation delay**：目前只模擬 queuing delay，可以再加上 `bw_r.txt` 第三欄的傳播延遲
2. **Per-link adaptive queue sizing**：不同 capacity 的 link 用不同 buffer size（like BDP-based sizing）
3. **EWMA smoothing**：如果 sim 振盪太劇烈，可以對 queue 狀態做指數移動平均
4. ~~**Sim 初始 queue 預填**~~：✅ 2026-04-13 校正為 1 step pre-fill（只用 `rollout_full` 內建那 1 步，不額外補）。原 2 步 pre-fill 造成 step 0 +674 ms 偏高；1 步對齊後 step 4 差僅 +0.7 ms
5. ~~**DRL loop double-advance 修正**~~：OSPF 模式已用 `--ospf` flag 修正（1 advance/step）。DRL 模式的 double-advance 作為 control delay 近似，暫保留
6. ~~**overhead_factor 校正**~~：✅ 2026-04-14 從 1.0288 校正為 1.032（經驗值，對齊 real 73 pkts/step fill rate）。Slope 誤差從 -6% 降到 +0.1%
7. ~~**queue_max_pkts + avg_pkt_bytes 校正**~~：✅ 2026-04-14 改為 `avg_pkt_bytes=1488`（L3 skb size）+ `queue_max_pkts=1000`（直接對應 netem limit）。Saturation delay 誤差從 -3.9% 降到 +0.07%。用 1488B (L3) 而非 1502B (L2)，因為 tc htb 計量不含 Ethernet header
8. ~~**獨立 TM 驗證**~~：✅ 2026-04-14 用 TM 00/01/02（SP MLU > 1.0 的壅塞 TM）獨立驗證。r² ≥ 0.997, bias < 3 ms，確認校正不是 overfit TM03
