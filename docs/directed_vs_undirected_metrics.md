# 鏈路指標設計問題：有向 vs 無向 — 無向指標如何系統性遮蔽非對稱壅塞

## 1. 有向 vs 無向 Link Utilization 定義

Full-duplex Ethernet 的每條物理鏈路天生就是**雙向獨立**的，即 A→B 與 B→A 各自擁有獨立的容量 C，互不干擾。OpenFlow 的 port statistics（tx_bytes, rx_bytes, tx_packets, rx_packets）也是逐 port、逐方向回報的，因此**原始量測資料本身就是 directed（有向）的**。

我們 codebase 中的實作錯誤地將兩個方向合併計算，形成 undirected（無向）指標：

以下令 speed 為單位時間內的傳輸速率（bits/s），C 為鏈路單向容量（bits/s）：

| | 有向（Directed，正確） | 無向（Undirected，legacy 錯誤設計） |
|---|---|---|
| 容量 | C | 2C（將雙向容量視為一體） |
| 觀測速率 | speed_tx = Δtx_bytes × 8 / Δt | speed = (Δtx_bytes + Δrx_bytes) × 8 / Δt |
| 利用率 | speed_tx / C | speed / 2C |
| 剩餘頻寬 | C − speed_tx | 2C − speed |

Legacy 實作中，link 層級的 free BW 還會對兩端 port 取 min（取最保守的一端）：

```
free_bw_port_i = 2C − speed_i    # switch A 端
free_bw_port_j = 2C − speed_j    # switch B 端
link_free_bw   = min(free_bw_port_i, free_bw_port_j)
```
兩端 port 量到的 speed 可能略有不同，但差異的來源是 **polling 時間差**，而非壅塞所致的鏈路中段丟包
> legacy 實作透過 HTB 做 rate shaping，超出 link capacity 的封包在進入 tc qdisc 前（或在 pfifo/netem enqueue 時）就被丟棄，這些丟棄**不會被計入發送端 port 的 `tx_packets`**；加上 veth pair 本身無損，因此 A 端 tx 與 B 端 rx 在統計上幾乎一致，僅有極少數意外 loss 與量測採樣不對齊造成的細微差異（詳見 [`loss_measurement_issues.md`](loss_measurement_issues.md) 第 3–5 節對 tc qdisc drop 路徑的分析）。

但無論取哪端，每端 port 的利用率實質上就是**兩個方向利用率的平均**：

```
util_port = speed / 2C = (speed_tx(A→B) + speed_tx(B→A)) / 2C
         = ( util(A→B) + util(B→A) ) / 2
```

## 2. 問題：非對稱壅塞在無向指標下不可見

考慮一個 4 節點拓撲，所有邊容量 = 1 Mbps。
下圖標示流量矩陣 TM、路由決策 decision，以及下發路由後有向圖上各邊的鏈路利用率。
![image](https://hackmd.io/_uploads/rky6aIsnbg.png)



| 指標 | 有向 | 無向 |
|---|---|---|
| Link (A,B) 的 LU | **100%**（已壅塞） | (100% + 0%) / 2 = **50%**（看起來正常） |

**一條已經在丟封包的鏈路，無向指標卻報告 50% 利用率。無論是人為判讀或 RL agent 依據此指標做決策，都會誤判該鏈路仍有餘量，導致持續分配流量至已壅塞的方向。**

## 3. Clipping 使問題在無向指標下進一步惡化

物理鏈路每個方向的實際吞吐量最多到容量上限 C，超額的封包在進入 qdisc 時即被丟棄，因此 **觀測到的 speed 天然被 clip 在 C**：

- **有向**：speed_tx ≤ C → free_bw = max(C − speed_tx, 0)，下限為 **0**，忠實反映該方向已滿載
- **無向**：speed = speed_tx + speed_rx ≤ 2C → 當 A→B 滿載（speed_tx = C）、B→A 空閒（speed_rx = 0）時，free_bw = max(2C − C − 0, 0) = **C**，仍顯示整條鏈路還有一半的剩餘頻寬

這意味著在無向指標下，**單方向超載的嚴重性被反方向的空閒容量稀釋了**：

### 具體數值

對任一 link (A, B)，容量 C：

| 方向 | 流量 | 真實利用率 | Clip 後剩餘頻寬 |
|---|---|---|---|
| A→B | 1.2C（超載） | 120% → clip 至 100% | **0** |
| B→A | 0 | 0% | **C** |

- **有向 MLU** = max(100%, 0%) = **100%**（已壅塞）
- **無向觀測剩餘頻寬** = max(2C − C − 0, 0) = **C**（觀測利用率僅 **50%**）

Agent 學到的規律：**「把流量塞進已壅塞的 link 不會有額外懲罰（被 clip），而且只要讓反方向少塞一點，就能進一步稀釋懲罰。」**

### 退化策略

在無向 MLU 優化下，agent 會收斂到：

1. **集中流量於單一方向**（因為反方向會稀釋利用率）
2. **容許單一方向達到或超過容量**（反正 clip 後損失有限）
3. **報告看起來健康的無向 MLU（~50%）**，但實際上封包正在丟失

## 4. 有向指標如何修正此問題

使用有向指標時，RL agent 觀測的是**每個方向獨立**的利用率：

```
state:  [free_bw(A→B), free_bw(B→A), ...]    # 每個方向獨立
reward: -max( util(A→B), util(B→A), ... )      # MLU = 最差的單一方向
```

性質：
- **無資訊損失**：非對稱壅塞直接反映在 state 中
- **無 clipping 漏洞**：超載任一方向都會被立即懲罰，不受反方向影響

## 5. 三個指標都受影響（BW / Delay / Loss）

有向 vs 無向的區別適用於 RL agent 使用的所有鏈路指標：

| 指標 | 無向（legacy） | 有向 | 使用無向的後果 |
|---|---|---|---|
| **剩餘頻寬** | 2C − (tx+rx) | C − tx | 遮蔽、誤判壅塞現象 |
| **延遲** | (d_LLDP(A→B) + d_LLDP(B→A)) / 2 | d_LLDP(A→B)（扣除 controller echo 後的單向 LLDP delay） | 遮蔽、誤判 delay |
| **丟包率** | max(loss(A→B), loss(B→A)) | 每方向獨立 loss(A→B) | 誤以為無壅塞邊壅塞 |

> LLDP delay 本身還有取樣偏差等系統性問題，dir/undir 公式的推導與偏差量化詳見 [`delay_measurement_issues.md`](delay_measurement_issues.md)。

## 6. 總結

| | 無向 | 有向 |
|---|---|---|
| 能偵測對稱壅塞 | 可以 | 可以 |
| 能偵測非對稱壅塞 | **不能** — 被平均掉 | **可以** |
| Clipping 產生漏洞 | **是** — 反方向吸收懲罰 | **否** |
| RL reward 忠實反映真實網路狀態 | **否** | **是** |

**結論**：基於 MLU 的 RL 目標必須使用有向（per-direction）鏈路指標。使用無向指標會系統性低估非對稱負載鏈路的壅塞程度，且產生的 reward signal 會引導 agent 收斂至病態的路由策略。
