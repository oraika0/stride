# DRL-OR-S 真實環境訓練流程分析

## Abstract

DRL-OR-S（TPDS 2023, Liu et al.）宣稱能對每個新來的 flow request 做 online routing。本文檢視其原始碼後釐清：它的「每條 flow 算一次路」並非由 first-packet 觸發的 reactive 決策，而是 **訓練腳本先合成 flow request，RL agent 在封包出現前就拿到決策所需資訊，算完路徑後先把對應 flow rule 下發到 controller，最後才在 host 上 spawn UDP server/client 把流量灌進去**。我們原本想做的「first packet 進 switch → packet-in 上送 controller → RL 決策 → 下發 flow rule → 封包繼續走」這條 reactive critical path（必須壓在 ~5ms 內），在 DRL-OR-S 的 testbed 流程中根本沒被觸發，因此 5ms 限制對它不適用——它是用 proactive 安裝規則的方式避開這個問題，代價是無法處理「不可預知的新 flow」這類真正的 reactive 場景。

---

## 一、DRL-OR-S 背景設定

為了避免之後混用詞彙產生誤會，先把 DRL-OR-S 在論文裡用的詞列出來，後面分析會沿用：論文講的是 `flow request`、`coming/upcoming flow`、`compute a route for each flow request`、`hop-by-hop route generation`。

幾個跟我們研究方向對照的設計差異：

- **Traffic 抽樣粒度（multi flow per demand）**：每個 OD demand 會被拆成多條獨立的 flow request（不同 5-tuple），每條 request 各自跑一次 RL 拿到自己的 path。對照我們的 single flow per demand（一個 OD demand 對應一個 routing decision），DRL-OR-S 的問題設計與決策粒度都比我們細
- **Multi-agent、next-hop decision**：不是一個中央 agent 直接輸出整條 path，而是每個 node 各部署一個 agent；對某條 flow request，從 src 開始一個 agent 決定下一跳、跳到下個 node 後該 node 的 agent 再決定下一跳，逐 hop 串成完整 path
- **DiffServ 4 種 traffic class**（rtype 0~3）：分別代表 delay-sensitive / throughput-sensitive / delay-throughput-sensitive / delay-loss-sensitive，每類有自己的 demand 大小與 reward 組成
- **流量產生**：testbed 不用 iperf，而是論文自己寫的一組 UDP client/server（細節見第四章末段）
- **演算法**：PPO + GCN encoder

---

## 二、背景問題

我們研究的方向是 reactive first-packet SDN routing：當新 flow 的第一個封包進入 switch、透過 packet-in 上送到 controller（KDN 的 Python 層）做決策再下發 flow rule，這條鏈路若超過約 5ms 就會失敗。本報告檢視 DRL-OR-S 在 Mininet+Ryu 上的實作，釐清它是否壓在這個時間預算內。

**結論先講：DRL-OR-S 並未壓在 5ms 內，而是透過 proactive 安裝規則的方式繞過這個限制。**

---

## 三、實作架構

DRL-OR-S 由三個獨立 process 組成，DRL agent 同時與 testbed 和 controller 各拉一條 TCP socket：

```
testbed.py (Mininet)     ryu controller.py     drl-or-s/main.py + simenv.py
   port 5000  ◄─────────────────────────────────  drl agent (主動 connect)
                           port 3999  ◄─────────  drl agent (主動 connect)
```

- **testbed.py**：建立 Mininet 拓撲，接收 agent 指令後在指定 host 上 spawn UDP server/client，回報量測結果
- **controller.py**：Ryu controller，接收 agent 指令後在指定 switch 上下發 flow rule
- **drl-or-s/simenv.py**：RL agent 端，產生 flow request、跑 policy 算路徑、同時驅動上面兩個 process

---

## 四、一個 time_step 的完整流程

每個 RL `time_step` 處理一條新 flow request——從合成、決策、裝 rule、起 UDP traffic、拿量測回饋，整個來回。最關鍵的一點是：

> **Flow 不是被偵測到的，而是訓練腳本自己合成出來的；產生後 RL agent 在封包尚未送出前，就已知道這條 flow 的 src/dst、demand、QoS class、duration。**

整個 pipeline 順序：

```
[訓練腳本合成 flow request] → [agent 取得 request 狀態] → [RL 算出路徑] → [裝 flow rule] → [實際送 traffic]
```

封包永遠是「rule setup 之後」才出生的。

### Step 0：訓練腳本合成下一條 flow request

`simenv._update_state()` 在每個 step 結尾抽樣產生下一條 request：用 demand matrix 抽 (src, dst)、抽 DiffServ class、抽 demand bandwidth、抽持續時間。

### Step 1：Agent 決策路徑（hop-by-hop）

從 src 開始，當前 node 上的 agent 看到 request + 區域觀察 → 輸出下一跳；換到下個 node 上的 agent 再決策一次，直到走到 dst。最終得到 `path = [s, ..., t]`。

### Step 2：Agent → Controller，預先安裝 flow rule

```
Agent  ── JSON {path, ipv4_src, ipv4_dst, src_port, dst_port} ──►  Controller (port 3999)
                                                                        │
                          沿 path 在每台 switch 下 priority=5 規則        │
                          match: 5-tuple (src/dst IP+port, ip_proto=17) │
                          action: OFPActionOutput(out_port)             │
                                                                        ▼
Agent  ◄───────────────  "Succeeded!"  ───────────────────────────  Controller
```

### Step 3：Agent → Testbed，現在才產生實際流量

```
Agent ── JSON {src, dst, src_port, dst_port, rtype, demand} ──►  Testbed (port 5000)
                                                                      │
        testbed.generate_request():                                   │
          1. dst_host.popen("server.py …")    ← 啟動 UDP server        │
          2. time.sleep(0.1)                  ← 等 server 起好         │
          3. src_host.popen("client.py …")    ← 開始送封包             │
          4. pmonitor 收 delay/throughput/loss 的第二筆 record         │
                                                                      ▼
Agent ◄───────── {delay, throughput, loss} ──────────────────────  Testbed
```

→ 計算 reward → 回到 Step 0 合成下一條 flow。

### 關於 UDP 流量與量測方式

- **不用 iperf 的原因**：iperf 主要量 throughput，沒辦法給 per-packet 的 one-way delay。DRL-OR-S 自己寫了一組 UDP `client.py` / `server.py`，**每個 UDP packet 內塞 sequence number 和送出 timestamp**，server 收到後直接算單向 delay，再用 sequence number 推估 loss。
- **量測時間很短**：server 每收到一小批 packet（rtype 0 是 10 個、其他 30 個）就 print 一次量測結果；testbed 只取**第二筆** record 當 reward（第一筆當 warmup 丟掉），實測約 100~200ms。所以這不是 flow 全長的統計，而是開頭的取樣回饋。
- **Flow 怎麼結束**：client/server process 本身不會自然停。Testbed 另外記一個 priority queue，等訓練邏輯時間走到該 flow 的 duration 結束點時，才主動 kill 掉 client/server。

### Test mode 也跑同一條 pipeline，這不是訓練技巧

`main.py` 的 train / test 兩個模式都會進入 `envs.step(actions, gfactors)`，預設 `simenv=True`，最後同樣呼叫 `sim_interact()` 走「先告訴 controller 裝 rule、再叫 testbed 起 UDP traffic」的流程。Train 跟 test 的差別只有兩點：

1. Test 把 actor 的 `deterministic` flag 開起來（關掉探索）
2. Test 不更新 policy gradient

`envs.reset()` 跟 `_update_state()` 也不分模式，都走第四章 Step 0 那段「訓練腳本自己合成 flow request」的邏輯。**沒有任何「test 模式改用真實封包觸發決策」的分支**——換句話說「flow 出現前就拿到 5-tuple 與 demand」這個 proactive 前提是寫死在架構裡的，不是訓練專屬的便利。

直接的推論：**論文 Section V Performance Evaluation 量到的 latency / throughput / loss，全部都是在 proactive pipeline 下取得的**。首包不會撞 packet-in、agent inference 不在 critical path 上，所以這些數字不能拿來支持「DRL-OR-S 在 reactive first-packet 場景下也能達到 XX 延遲 / XX QoS」的推論。

---

## 五、為什麼叫 proactive、為什麼沒有 5ms 限制

**Proactive vs reactive 指的是 flow rule 安裝時機相對於封包的關係。**

### Reactive（我們想做的方向，5ms 限制就在這裡）

```
封包先到 switch ── packet-in ──► controller ── RL 決策 ──► 下發 flow rule ──► 封包繼續走
                                  ↑ 這整段必須 < 5ms，不然首包 timeout
```

封包**先到**，rule **後裝**。

### Proactive（DRL-OR-S 的做法）

訓練腳本**先**給 agent 完整 request，agent 算完 path 之後 controller 對所有 path 上的 switch 預先下發 rule，**再**起 server/client 灌流量。第一個封包進入 s1 時，flow table 早有對應的 priority=5 entry，連 packet-in 都不會觸發。資料面首包不承擔任何 RL inference 或 rule setup 的延遲，5ms 預算對它不適用。

### 論文數字佐證

論文 Section V-A Experiment Setup 寫：

> "In our experiments, **an agent needs only 3 ms to select the next hop** given an input state."

這 3ms 是 **per-node、per-hop 的 forward time**。DRL-OR-S 是 multi-agent hop-by-hop sequential（一個 agent 跑完才換下一個），所以一條 flow 的總決策時間 ≈ `hop_count × 3ms` + controller socket round-trip + 各 switch 下發 flow rule。**只要 hop count 大於 2 就已經超過 5ms reactive 預算**，再加 socket 跟 rule 下發時間更不可能。論文自己的數字就證明它若直接搬到 packet-in critical path 上，多 hop flow 會超時。

---

## 六、這算真實場景嗎

DRL-OR-S 的整個流程建立在「flow 出現前就能拿到 5-tuple、demand、QoS class、duration」這個前提，所以能不能對應到真實場景，端看這個前提在該場景裡成不成立。

### 6.1 它無法處理的場景

| 真實場景 | DRL-OR-S 為什麼無法處理 |
| --- | --- |
| 使用者開瀏覽器、TCP SYN 直接打進 switch | 無法在 5-tuple 出現前事先取得；即使硬要當下決策，per-hop forward 自報就 3ms，多 hop 一定超過 5ms 預算 |
| 大量短 mice flow（< 1s）湧入 | 無法處理「flow 比決策 + rule setup overhead 還短」的情形——rule 才剛生效，flow 已經結束 |

### 6.2 它能處理的場景

只要在 flow 出現前，已能取得 5-tuple 與 QoS 需求，DRL-OR-S 的設計就能對得上：

| 真實場景 | DRL-OR-S 為什麼能處理 |
| --- | --- |
| SD-WAN site-to-site tunnel、VoIP call setup | Signaling 流程在實際送資料前就協商好 5-tuple 與 QoS 需求 |
| ISP elephant flow、頻寬預約 | 客戶在開通前就跟 operator 約定好兩端 endpoint 與容量需求 |

### 6.3 小結

DRL-OR-S 確實做到「對每個 flow request 算一條路」，決策粒度比 OD-level / epoch-level 細。但它能跑的前提是「flow 出現前資訊就齊備」，因此它的適用範圍只涵蓋預先協商或預約類的 proactive 場景，而把 reactive SDN 最痛的那塊（5-tuple 不可預知 + 5ms 決策預算）排除在外。

---

## 七、結論

DRL-OR-S 看起來像 reactive online routing，但 Mininet+Ryu testbed 的實作顯示它其實是 proactive：先合成 flow request、先裝 rule、再灌流量。而且 train 跟 test 走的是完全同一條 pipeline，並沒有「測試時切換成真實封包觸發決策」的設計。我們原本卡的 5ms first-packet critical path 在它的流程中沒被觸發，所以它的「per-flow online routing」與我們想做的「first-packet 觸發 RL」不是同一個問題；論文報的 testbed 評估數字也只反映 proactive 場景下的表現，無法外推到 reactive 場景。
