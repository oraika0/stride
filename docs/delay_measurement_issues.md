# Per-link delay 量測問題：LLDP 系統性低估真實 queuing delay，改用 tc backlog 作為 ground truth

> 本文檔記錄了我們在 Mininet 環境中測量 per-link directed delay 的完整分析：
> 從最初發現 LLDP delay 嚴重低估，到用 tc backlog 和 UDP probe 交叉驗證，
> 量化了 LLDP 的系統性偏差（穩態下仍約為真實 delay 的 1/3），
> 並釐清了 netem queue backlog 在非壅塞鏈路上的 burst transient 現象。

---

## 0. 方法來源與修改

我們拿來驗證的 delay 量測工具 **(UDP-Probe)** 來自 DRL-OR-S 的原始實作，原版只產出 **per-path delay**（供 agent state / reward 使用），沒有 per-link 的 raw LLDP 輸出。

本研究的需求：
1. **Agent 需要 per-link 指標**：在我的實驗環境中，需要的是每條 link 的 delay / loss / util，而不是路徑聚合值。
2. **確認 per-link 量測本身正確**：必須先檢查 link-level 原始值是否反映真實 queuing delay。

**本文的工作 = 把 UDP-Probe 的輸出改成 per-link（directed），並用 tc backlog、UDP probe、LLDP 交叉比較以驗證其正確性**。結論會發現 LLDP-based 的 per-link delay 有系統性偏差，這就是下一節要釐清的核心問題。

---

## 1. 問題起源

在 test_single_tm 的 eval_metrics 中觀察到：

- **TM 03 steady state avg_delay ≈ 38 ms** — 看起來不大
- 但這是 37 條 link 的平均值，其中只有 **link 13→14（1.55 Mbps）** 真正壅塞
- 13→14 穩態下 LLDP directed 讀到 ≈ **2700 ms**（已被平均稀釋到 38 ms）
- 同一時刻 tc backlog 反算的 queuing delay 是 **~7700 ms**（UDP probe 也是 ~7700 ms）— 差了約 3 倍

**tc backlog 反算 queuing delay 的公式：**

```
queuing_delay = backlog_bytes × 8 / C
```

backlog_bytes 是 `tc -s qdisc show dev` 回報的 netem 當下 queue 佔用量，C 是該 link 的 bandwidth（bps）。以 13→14（C = 1.55 Mbps, Q 達 limit=1000 packets, 每 packet ≈ 1488 B）為例：

```
1000 × 1488 × 8 / 1,550,000 ≈ 7.68 s
```

這是 queue 現時佔用所對應的排隊時間，物理意義明確、不受取樣偏差影響，可當 ground truth。

這個 LLDP vs tc 的 3 倍差距是系統性的還是偶發的？需要釐清 LLDP delay 的測量機制。

### 1.1 延遲組成：本研究量測的是哪一項

單跳 link 的 end-to-end delay 由四項組成，四項在真實網路中都存在。以 1.55 Mbps / 1488 B pkt 為例：

| 延遲類型 | Mininet 典型量級 | 說明 |
|----------|------------------|------|
| **Queuing**（排隊，`q_bits / C`） | 0 – 7.68 s（依 backlog） | 壅塞時絕對主導項 |
| **Transmission**（傳送，`pkt/C`） | 7.7 ms / pkt | 壅塞時被 queuing 淹沒（差 1000×）|
| **Propagation**（傳播） | veth ≈ 0 | Mininet 在沒設定delay時 ≈ 0 |
| **Processing**（處理） | OVS fast-path μs 級 | 量級太小可忽略 |

三種量測法的涵蓋範圍並不一樣：**tc backlog** 只讀 netem 當下的 queue 佔用，只量到 Queuing；**UDP probe fwd delay** 和 **LLDP directed** 是 end-to-end 量測，物理上包含全部四項。但因為 Mininet 下後三項都接近 0（transmission 在壅塞時被淹、propagation 為 0、processing μs 級），probe / LLDP 的讀值非常接近 queuing — 後續在此化簡下使用 queuing 作為delay 分析。

> 目前 delay 訊號只取 tc backlog 反算的 queuing，其餘三項未納入。未來可以補上 transmission floor（`delay = q/C + pkt/C`，`pkt/C` 為 per-link 常數）、propagation delay（依實驗設計，e.g. 5 ms）、processing 忽略。

---

## 2. LLDP Delay 的兩個公式

LLDP delay 不是直接量 A→B 的 one-way link delay，而是 **controller 透過 OpenFlow 送一個 LLDP frame 經由 switch k 出 link 抵達 switch l，再由 l packet_in 回 controller** 所觀察到的 round-trip 時間扣掉 channel 成分：

![image](https://hackmd.io/_uploads/ByrQ8jhhbg.png)

如圖所示，raw LLDP delay 包含三段：
- **controller ↔ switch *k*** 的 OpenFlow channel 延遲（用 echo message 量得的 `echo_src`）
- ***k*→*l* 真正的 link traversal delay（我們想要的那段）**
- **switch *l* ↔ controller** 的 OpenFlow channel 延遲（`echo_dst`）

`simple_delay.py` 中有兩個公式處理這筆原始值：

### 2.1 Undirected：`get_delay()`

`get_delay(src, dst)` 把兩方向 raw LLDP 相加後扣雙邊 echo RTT 取半：

```python
fwd_delay = graph[src][dst]['lldpdelay']   # raw LLDP A→B
re_delay  = graph[dst][src]['lldpdelay']   # raw LLDP B→A
delay = (fwd_delay + re_delay - src_latency - dst_latency) / 2
```



### 2.2 Directed：`get_link_delay_dir()`

```python
lldp_fwd = graph[src][dst]['lldpdelay']          # raw LLDP fwd
echo_src = echo_latency[src]
echo_dst = echo_latency[dst]
d_dir = max(lldp_fwd - echo_src/2 - echo_dst/2, 0)
```

只使用 forward 方向的 raw LLDP，並扣除 controller ↔ switch 兩側的 echo RTT/2，還原 A→B 的單向估計。

---

**後續章節的分析一律以 directed delay（§2.2）為準**；undirected 的偏差問題請見 [`directed_vs_undirected_metrics.md`](directed_vs_undirected_metrics.md) §5。本文接下來探討的是：即使用 directed 公式，LLDP 仍為什麼會系統性低估真實 queuing delay。

---

## 3. UDP Probe + tc Backlog 驗證實驗

為了量化 LLDP 的偏差程度，我在 `test_single_tm_udp.py` 中同步採集三種 delay 來源：

### 3.1 三種測量方式

| 來源 | 方法 | 時效性 |
|---|---|---|
| **tc backlog** | `tc -s qdisc show dev` 讀取 netem 的 `backlog` 計數器，再除以 link bandwidth 得到 queuing delay | 當前瞬時值 |
| **UDP probe** | 從 src host 發 UDP echo packet，server 端回傳時戳，計算 forward one-way delay | 該 probe 封包實際穿越 link 所經歷的 delay |
| **LLDP** | Ryu controller 注入 LLDP frame 到 switch，測量穿越 datapath 的時間 | 最近一次 LLDP frame 的結果（每 8 秒更新一次，`DELAY_DETECTING_PERIOD`） |

---

## 4. 實驗結果

### 4.1 單一 TM 的壅塞鏈路時序觀察（link 13→14, 1.55 Mbps, TM 03）

![timeseries_03_13_14](https://hackmd.io/_uploads/S1isN3h3be.png)

圖中四條 delay 軌跡：
- **tc backlog**（黑，基準）與 **UDP probe fwd**（橘）幾乎重疊，從 step 0 線性爬升，step 12 之後達到 7700 ms 並鎖在該值 — 對應 Q=limit=1000 的 tail-drop 穩態（§1 公式）
- **LLDP fwd (directed)**（紫紅）同期爬升，但step9發生突降，穩態僅停在 ~2700 ms
- **LLDP undirected**（青）更低，穩態 ~1300 ms 

tc 與 UDP probe 一致是合理的 — UDP probe 封包就是跟著 data 流走同一條 netem queue。奇怪的是 LLDP：它也該經歷同一條 queue，卻系統性只讀到 1/3 的值。

> 圖中 tc 軌跡偶爾瞬間掉到 0 不是 queue 真的清空，而是 `tc -s qdisc show dev` 與 kernel netem dequeue batch 之間的 race condition，§6 會詳細說明並給出 retry-on-zero 的修正。

### 4.2 跨 TM 一致性（link 13→14 over 5 test TMs）

![timeseries_5tm_13_14](https://hackmd.io/_uploads/HkJEH3n2Ze.png)

上四張：TM 00/01/02/03 — 13→14 被選為承載路徑、queue 飽和。tc/UDP 都到 ~7700 ms，LLDP directed 穩態都落在 ~2700 ms 左右。偏差比例在不同 TM 間高度一致（~35%）。

下四張：TM 10/12/14/21 — 13→14 未被選中、無壅塞。四條軌跡皆在 0~30 ms 區間抖動，彼此差異不具物理意義（屬 §7 的 burst transient）。

> 各 TM 之間 y 軸不從 0 開始，主要是因為 **DRL step 0 並不等於 traffic 啟動時刻** — 依 [`sim_fluid_queue_calibration.md`](sim_fluid_queue_calibration.md) 的時序，iperf3 clients 在 t=40s 啟動、而第一個 DRL step 在 t=60 才被記錄，queue 在 step 0 之前就已經開始累積。
> 
### 4.3 穩態（step ≥ 12）統計

針對壅塞 link 13→14 在 TM 03 steady state：

| 測量方式 | Steady-state | 佔 tc 比例 |
|---|---|---|
| **tc backlog** | ~7700 ms | 100%（基準） |
| **UDP probe fwd** | ~7700 ms | ~100% |
| **LLDP directed** | ~2700 ms | **~35%** |

### 4.4 Probe vs tc 相關性

| 範圍 | Pearson r | r² | N |
|---|---|---|---|
| 全部 link | 0.977 | 0.954 | 11,093 |
| 壅塞 link（tc > 100ms）| 1.000 | 1.000 | 26 |
| 非壅塞 link（tc ≤ 100ms）| 0.006 | 0.00004 | 11,067 |

N 是納入相關性計算的 (link, step) sample 數。理論總量 = 5 TM × 30 step × 74 directed link = 11,100，扣除 `probe_ok=False`（probe 封包遺失）後剩 11,093。

- **壅塞 link（r ≈ 1.000）**：queue 鎖在 limit=1000、backlog 幾乎是常數，tc 和 probe 都讀到同一個 ~7700 ms 的穩態值，兩者完美一致。
- **非壅塞 link（r ≈ 0）**：tc 和 probe 的讀值都落在 0–幾十 ms 的 burst transient 範圍內（§7），這些抖動來自 kernel NAPI/softirq 的 batch processing，tc 讀的是查詢當下的瞬時 backlog、probe 讀的是該封包實際穿過 queue 的體驗 — 兩個取樣時刻不同，batch 事件又是隨機發生，因此在低量級區域兩者是**各自獨立的雜訊**，相關性近乎 0。這不代表兩者中有一者錯了，反而印證 §7 的結論：非壅塞 link 上的 delay 本來就是不可預測的 burst 抖動，沒有共同的 latent ground truth 可以讓兩者齊漲齊跌。
- **全部 link（r ≈ 0.977）**：壅塞的少數點（~7700 ms）把整體相關性拉高，即使多數非壅塞點是雜訊，Pearson r 仍被大斜率那一段主導。

**結論：在有壅塞的區段，UDP probe 和 tc backlog 互為 ground truth；在非壅塞區段，兩者都只是在反映 burst transient 的隨機波動，沒有誰比較「對」的問題。LLDP 才是系統性的偏差來源（§5）。**

---

## 5. LLDP 為什麼系統性低估

### 5.1 穩態下 LLDP 仍只讀到 ~1/3，不是取樣偏差能解釋的

一個直覺的假設是「LLDP 每 8 秒才量一次，所以讀到的是 queue 尚未飽和時的舊值」。但這個假設在 **穩態**（§4.1 圖 step ≥ 13）下不成立：

- queue 已達 limit=1000 並在 tail-drop，backlog 鎖在最大值，**queuing delay 對時間幾乎是常數 ~7700 ms**
- 不論 LLDP frame 何時進 queue，在 FIFO + 飽和 backlog 的前提下都該排在 999 個 packet 之後 → 應經歷 ~7.7 s 的排隊時間
- 但實測 LLDP 穩態讀值鎖在 ~2700 ms，完全不接近 tc/UDP 的 7700 ms

取樣偏差（只在 ramp-up 期才會造成「讀到歷史值」的錯覺）不能解釋這 3 倍 gap。**實際機制我們尚未查明**，可能的方向：

- OVS 對 controller 注入的 packet_out（LLDP frame 的送出路徑）走不同的 priority class，未與 user data 共用同一條 tc qdisc/netem buffer
- 或 LLDP frame 進入 switch 後並非經由 netem 的 FIFO queue 送出 — OpenFlow `packet_out` 的 action path 是否實際經過 egress netem 需要 tracing 驗證

這個偏差在不同 TM、不同 steady-state step 下都穩定在 ~35%（§4.2），是**系統性**而非抖動。

> Undirected 公式 `(fwd+rev)/2` 會再把這個已經偏低的 LLDP 值進一步稀釋，但這是「有向 vs 無向」指標設計層面的議題，詳見 [`directed_vs_undirected_metrics.md`](directed_vs_undirected_metrics.md) §5，本文不重複。

---

## 6. tc Backlog 的 Kernel Race Condition

### 6.1 現象

在少數情況下，`tc -s qdisc show dev` 讀到 `backlog=0`，
但同一時刻 UDP probe 的 fwd delay 顯示 queue 明顯非空（> 50ms）。

### 6.2 原因

`tc -s qdisc show dev` 讀的是 netem qdisc 裡的 `qstats.backlog` 欄位 — 這個欄位是 kernel 追蹤「目前還在 queue 內的 bytes 數」的計數器，`enqueue` 時 `+= skb->len`、`dequeue` 時 `-= skb->len`。

問題出在 netem 的 dequeue 路徑是**non-atomic的**：

```
softirq → qdisc_run → netem dequeue_skb
    ① 從 internal rb-tree 取出一個 skb（時間到期的封包）
    ② qstats.backlog -= skb->len         ← 計數先減
    ③ 把 skb 交給下一層（driver / 下游 qdisc）
    ④ 回到 ① 繼續取下一個 skb（同一次 softirq 可能連續 dequeue 多個）
```

如果 queue 裡只剩少量封包、而 user-space 的 `tc -s` 系統呼叫恰好落在步驟 ②–④ 之間，kernel 會回報「目前 backlog 已扣到 0，但正在處理的這個 skb 還沒真的送出、也還沒有新封包 enqueue 上來」的中間狀態 — `tc` 於是讀到 `backlog=0`，即使從使用者的角度看 queue 根本沒空。

這不是 counter 的 bug，而是 non-atomic 讀取的正常結果：`tc` 是事後查詢工具，沒有對 qdisc 加鎖，拍到 dequeue batch 中間的一瞬是合理的。

### 6.3 發生率

分母必須侷限在「probe fwd 已經明顯不為 0」的 samples（代表 queue 確實有 backlog，此時 tc 讀到 0 才能被判定是 race），否則用全部 sample 當分母會被大量非壅塞、本來就該是 0 的樣本稀釋而失真。

以 `probe_vs_tc.csv`（5 個 test TM、已套用 retry=3）中 `probe_fwd_ms > 100` 的 samples 為分母：

| retry 策略 | Race condition 比例 |
|---|---|
| retry=3（現行） | 1/27 ≈ 3.7%（`probe_fwd_ms > 100` 且 `tc_delay_ms = 0`） |
| 無 retry | 尚未重跑量化，留待之後補上對照 |

### 6.4 修正

`tc_stats_dev()` 加入 retry-on-zero 邏輯：

```python
def tc_stats_dev(dev, retries=3):
    for _ in range(1 + retries):
        sent, dropped, backlog_bytes, backlog_pkts = _tc_stats_dev_once(dev)
        if backlog_bytes > 0:
            return sent, dropped, backlog_bytes, backlog_pkts
    return sent, dropped, backlog_bytes, backlog_pkts  # 真的是 0
```

trade-off：每次 retry 是一次 `subprocess.check_output("tc -s qdisc show dev ...")`，延遲約 3-5ms。
在 backlog 真的為 0 的情況下（絕大多數），retry 3 次的額外開銷 ≈ 10-15ms per link per step。

> 更精確的方法是使用 kernel netlink socket 直接讀取 qdisc stats（`pyroute2` 或 `libnl`），
> 避免 fork subprocess，但對現有 Mininet 框架改動較大，目前 retry 已足夠。

---

## 7. netem Queue 在非壅塞鏈路上的 Burst Transient

### 7.1 現象

即使沒有超額流量（free_bw > 0），tc backlog 偶爾讀到非零值：

| 條件 | 非零 tc 比例 | tc mean (when > 0) | tc max |
|---|---|---|---|
| bw ≥ 25 Mbps, steady state | 36.3% | 0.37 ms | 1.91 ms |
| bw = 1.55 Mbps, 低負載 link | ~50% | ~6 ms | ~15 ms |

### 7.2 這不是 bug — 是 kernel batch processing 的正常行為

封包在到達 netem qdisc 之前經過多層 kernel 處理：

```
OVS kernel datapath
    → NAPI polling（NIC driver 的 batch 收集）
    → softirq 處理（一次處理多個 skb）
    → dev_queue_xmit（送入 tc qdisc）
```

每一層都有 **batch processing**：NAPI 一次 poll 可收集多個封包，softirq 一次處理多個 skb，
OVS datapath 也是 batch forward。這意味著即使流量的平均速率低於 link bandwidth，
封包也不是均勻地一個一個到達 netem 的 — 它們以 **burst** 的形式到達。

### 7.3 數學解釋

假設一條 25 Mbps link 上有 10 Mbps 的流量（util = 40%）。

- 平均 packet inter-arrival time = 1500 × 8 / 10M = **1.2 ms**
- 但 kernel batch 一次送 5 個 → burst interval = 0，之後 idle 6ms
- burst 瞬間：5 個封包同時 enqueue 到 netem
- netem 以 25 Mbps dequeue：每個封包 1500 × 8 / 25M = **0.48 ms**
- burst 的最後一個封包需等 4 × 0.48 = **1.92 ms**
- 然後 queue 清空，直到下一個 burst

**這就是為什麼 25 Mbps link 上 tc 讀到 max ~1.91 ms — 恰好是一個 burst 的排隊時間。**

### 7.4 對 1.55 Mbps link 的影響

同樣的 burst size，在 1.55 Mbps link 上排隊時間更長：

- dequeue time per packet = 1500 × 8 / 1.55M = **7.74 ms**
- 5-packet burst 的最後一個等 4 × 7.74 = **30.97 ms**

這解釋了為什麼低頻寬 link（如 3→14, 1.55 Mbps）即使不壅塞也能看到 ~15ms 的 tc 值。

**結論：burst transient 是真實的物理現象，不需要 filter。** 它們代表了封包在 qdisc 中的真實排隊體驗，
在用 delay 作為 RL state 時，保留這些資訊是正確的。

---

## 8. 目前實驗改動

### 8.1 UDP probe 的定位：目前僅作為驗證工具

UDP probe 本次引入的目的是**驗證 tc backlog 反算 queuing delay 的正確性** — 讓一個實際穿越 queue 的封包提供 ground-truth 對照。§4.4 已證實 probe fwd 與 tc 相關性 r=1.000（壅塞段），兩者在壅塞區段互為真值。

UDP probe **暫時不進入** RL 實驗的常駐量測迴圈，主要卡點不在效能而在**路由耦合**：

- 目前的 per-link UDP probe 共用 agent 的 flow rules。現行實驗是 SP routing，在本拓撲下碰巧給每對相鄰 switch 產生 one-hop 的 shortest path，probe 封包因此能沿單一 link 直達、量到該 link 的 delay。
- 一旦切到 RL routing（或任何非 one-hop 的路徑選擇），相同 (src, dst) 的 probe 封包會被導到多跳路徑，量到的是 path-level delay 而不是 per-link delay。
- 要在常駐迴圈裡使用 probe，必須先設計**獨立於 agent routing 的 probe 專用 flow rules**（例如針對 probe 封包套用分離的 match field + dedicated path），這部分目前還沒有明確解法。

> tc backlog 作為 real-env delay 訊號的說服力目前**也還不夠充分**。原則上 end-to-end 量測（LLDP / UDP probe）才是封包實際經歷的延遲，從方法論角度也比讀 kernel 內部 counter 更合理 —— 若被追問「為什麼不用 end-to-end」，目前沒有好答案。現行選擇 tc 純粹是因為 LLDP 有 §5 尚未 pinpoint 的系統性低估、UDP probe 又因 §8.1 的路由耦合無法常駐。換言之 tc 是**繞開壞掉 / 難用的 end-to-end 量測的 workaround**，只涵蓋 queuing、且只在壅塞區段有 §4.4 的 r=1.000 佐證。長期仍應回到能常駐的 end-to-end 量測。

### 8.2 後續 RL 實驗：以 tc 取代 LLDP 作為 delay 訊號

| 層面 | 原先 | 修改後 |
|---|---|---|
| **Delay state / reward** | LLDP directed（穩態仍低估 ~3×，且 8s 週期滯後） | tc backlog 即時反算的 directed queuing delay |
| **Ground-truth 驗證** | 無 | UDP probe 離線 spot-check（僅在驗證性實驗中採集，不進常駐訓練 / 評估 loop） |
| **LLDP** | state / reward 主要來源 | 保留供參考，不入 agent observation |

---

## 9. 工具與腳本

| 工具 | 用途 | 位置 |
|---|---|---|
| `test_single_tm_udp.py` | DRL eval + 同步 tc/probe/LLDP 採集 | repo 根目錄，仍可重跑 |
| 繪圖腳本 | 繪製 scatter、timeseries、per-TM table | 已於 2026-08 移除 |
| `diagnostics/udp_probe_echo.py` | UDP echo server/client | `diagnostics/` |
| 實驗數據 | 5 個 test TM 的 `probe_vs_tc.csv` | session `20260413_130143` 已移除，本文表格為當時結果 |
| 圖表 | scatter、timeseries、per-TM table SVG | 已隨腳本移除 |

---

## 10. 總結

穩態下（壅塞 link 13→14, Q 達 limit=1000, tc backlog 鎖在 7700 ms）各量測來源讀值：

```
tc backlog:      真實 queuing delay 的 100%   ← 基準（backlog × 8 / C）
UDP probe fwd:   真實 queuing delay 的 ~100%  ← 封包實際體驗，與 tc 吻合
LLDP directed:   真實 queuing delay 的 ~35%   ← 系統性低估 3×（§5.1 機制未明）
```

**結論：目前異常的 LLDP delay 不適合作為 RL agent 的 state observation 或 reward signal — 即使用 directed 公式並扣除 echo correction，穩態下仍只讀到真實值的 ~1/3，且原因尚未驗證。先改用 tc backlog 計算的 directed queuing delay，輔以 UDP probe 做驗證。**

---


## 11. 未來工作：定位 LLDP 低估的根本原因

§5.1 列了兩個候選機制（OVS priority class / `packet_out` 繞過 egress netem），但都只是猜測，沒有 tracing 佐證。這條路不需要實作任何 probe infrastructure、純觀察 / tracing，且若 LLDP 能修就不必投入 §12，因此邏輯上應先做。

**驗證路徑（從便宜到貴）**：

1. **抓包驗證 LLDP frame 的實際路徑**：在 switch k 的 veth egress 用 `tcpdump` 觀察 LLDP frame 是否真的從 netem 出去、時戳和 raw LLDP delay 是否吻合。若抓不到，代表 LLDP 根本沒走 user data path。
2. **追 OVS datapath / kernel trace**：用 `ovs-appctl dpctl/dump-flows`、`perf trace` 或 `bpftrace` 看 `packet_out` 的實際 egress 路徑，確認是否經過 tc qdisc。
3. **閱讀 Ryu `switches.py` 的 LLDP 注入點**：確認 controller 側送出 LLDP 時用的 OpenFlow action（`OUTPUT:port` vs `OUTPUT:LOCAL` vs `OUTPUT:CONTROLLER`），以及 switch 側收到後的 `packet_in` 觸發路徑是否繞過 queue。

若機制可修（例如改 OpenFlow action、改注入點），直接修；若不可修（kernel fast path 設計使然），此路走不通，退回 §12 的 dedicated UDP probe 方案。

---

## 12. 未來工作：以 dedicated flow rules 讓 per-link UDP probe 可常駐

若 §11 的 tracing 確認 LLDP 無法修，改走這條路。核心是讓 probe 封包**不被 agent 的 routing flow rules 重寫**

### 12.1 設計

1. **Probe 識別**：保留 UDP port range `55000–55999`，port 號後三碼編碼目標 link（例如 `udp_dst=55013` 代表量 link 0→13）。封包 payload 帶 `t_send`。

2. **Dedicated flow rules**（`priority=100 > agent rule`）分兩類：

   ```text
   # Forward rule — 每條 directed link 一條，裝在 source switch：
   S_A: match udp_dst=55_AB                     → output port_to_B
   S_B: match udp_dst=55_BA                     → output port_to_A
   # ...共 74 條（每個 directed link 一條，A→B 和 B→A 不共用）

   # Delivery rule — 每個 host 一條，裝在其 attached switch：
   S_B: match udp_dst∈55000-55999, eth_dst=h_B  → output port_to_h_B
   # ...共 23 條（一個 node 一條，不分 incoming link）
   ```

  

3. **量測方式 — one-way**：每台 switch 掛 probe host（重用 `h_k`）。h_A 在 payload 寫 `t_send` 發出，h_B 收到記 `t_recv`，`d(A→B) = t_recv - t_send`。

   - **接入段（veth / NIC）修正**：原理上 raw = `veth_A + d(link) + veth_B`，應另跑 host↔switch loopback 量接入段再扣（結構同 LLDP 扣 `echo/2`）。但是目前LLDP echo 量級為0.x~1ms，遠低於queuing delay，暫時未實做

4. **分散式並發發送**：Geant 23 switches × 74 directed link → 每台 h_A 平均只發 ~3 個 outgoing probe。各 h_A 獨立 process 同 round 發送（需一個 orchestrator 發 trigger），各 h_B 本地收集再批次上傳 collector。避免單 process asyncio + GIL 的累積延遲。

### 12.2 風險與工程預算

| 項目 | 預算 / 限制 | 風險 |
|---|---|---|
| 單 round 總時間 | < 100 ms（確保 probe round 近似瞬時 snapshot） | 向 23 host 並發 trigger 的時序對齊精度決定下限 |
| 單 h_A 發送 ~3 probe | 3~6 個 `sendto()` 依序呼叫，整批 < 10 μs | 相對 ms 級 delay 可忽略 |
| 時鐘解析度（單 host） | `clock_gettime` μs 級抖動 | 遠小於 ms 級 queuing delay，非瓶頸 |

### 12.3 驗證順序

1. **Single-link spike**：先驗 dedicated flow rule 的 `priority > agent` 真的能讓 probe 無視 agent routing（這是整個方案的前提假設）。
2. **量 orchestrator trigger 的對齊精度**：23 個 h_A 同時收到 trigger 後發送 probe 的時間差要 ≪ queuing delay 本身的抖動幅度。
3. **全拓撲鋪 97 條 rule + 並發量測**。


