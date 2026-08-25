# Per-link loss 量測問題：netem/tc drop 計數路徑分析與 UDP probe 交叉驗證

> 本文檔記錄了我們在 Mininet 環境中測量 per-link packet loss 的完整歷程：
> 從最初的「loss 永遠趨近 0」到找出 tc qdisc 架構的根本問題，最終以 netem-only 方案根治。
> 同時對照真實 SDN ASIC 交換機的 buffer 架構，驗證我們的模擬是否合理。

---

## 1. 問題起源

在分析 Mininet 實驗的 directed link metrics 時，發現以下異常：

- 明顯壅塞的 link（free_bw ≈ 0、delay 高達數秒）卻顯示 **loss ≈ 0%**
- 在原始 Geant with TM*3 環境下，LP 求解的 optimal MLU(directed) = 1.2（即使在最佳 traffic splitting 下，最繁忙的 link 也有 120% 的 utilization），OSPF 只會更差
- queue depth（backlog）持續在 200~500 packets，明顯有壅塞，但就是沒有 drop

為了釐清這個現象，我從 Mininet 的封包路徑開始，逐層分析 OVS → kernel → tc qdisc 的完整處理鏈。

---

## 2. 封包在 Mininet 中的完整路徑

```
h_src (iperf3 UDP) → h_src-eth0 → [veth pair] → s_src-eth_host
    → OVS kernel module (flow table match → 選定 egress port)
    → s_src-eth_egress → [tc qdisc chain on egress port]
    → [veth pair] → s_dst-eth_ingress → OVS → ... → h_dst
```

### 2.1 名詞說明

上圖中的命名對應關係：

| 圖中名稱 | 意義 | Mininet 實際名稱 |
|---|---|---|
| `h_src` | Host（Linux network namespace，可跑 iperf3） | `h1`、`h2`… |
| `h_src-eth0` | Host 裡面的網卡介面（如同你筆電上的 `eth0`） | `h1-eth0` |
| `s_src-eth_host` | Switch 上**朝 host** 那一側的 port | `s1-eth1`（依 port number） |
| `s_src-eth_egress` | Switch 上**朝下一跳 switch** 的 port（OVS 查完 flow table 後選中的出口） | `s1-eth2`（依 port number） |

每條 Mininet link 都是一對 **veth pair**（虛擬雙頭網卡），兩端各放在不同的 namespace，效果等同一條網路線：

```
[h_src namespace]                  [s_src (OVS bridge)]
     │                                    │
 h_src-eth0 ◄════ veth pair ════► s_src-eth1 (port 1)
                                  s_src-eth2 (port 2) ◄════ veth pair ════► s_next-eth3
                                  ...
```

### 2.2 背景：qdisc 架構速查

**qdisc（queueing discipline）是以「NIC 的 egress 方向」為單位** — 每張介面的出方向掛一個 root qdisc，封包**離開**該介面時經過它。Ingress 方向預設不處理。

因此「link A→B 的 drop」意思是：**A 端 NIC 的 egress qdisc 丟棄了封包**，不是發生在 B 端。

qdisc 分兩類：

| 類型 | 特色 | 例子 |
|---|---|---|
| **classless（無類）** | 一個佇列，不分層 | `pfifo`、`netem` |
| **classful（有類）** | 底下可開多個 class，每個 class 再掛子 qdisc，形成樹狀結構 | `HTB`、`HFSC` |

Mininet 使用的 HTB 是 classful qdisc，其內部結構為三層：

```
NIC egress
 └── qdisc HTB (root)             ← 排程演算法，決定「何時讓誰送」
      └── class 1:1 (rate=X)      ← 一組限速規則（rate / ceil / burst）
           └── qdisc pfifo        ← 真正存放封包的 FIFO buffer（leaf）
```

| 層 | 職責 | 能不能存封包 |
|---|---|---|
| HTB (root qdisc) | 排程、限速演算法 | **不能** — 只是調度器 |
| class 1:1 | 速率帳戶（持 rate/ceil/burst 參數 + token bucket + 指向 leaf 的指針）| **不能** — 本身沒有 buffer |
| pfifo (leaf qdisc) | 真正的佇列 | **能** — 封包排隊在這裡，滿了就丟 |


Drop 永遠發生在 **leaf qdisc（pfifo 或 netem）**，因為只有 leaf 才有 buffer；class 和 root 都只做記帳和排程，沒有空間可丟。這點在下一節追蹤「計數器為什麼看不到 drop」時很關鍵。

當 Mininet 的 `addLink()` 帶了 `max_queue_size=Y` 時，kernel 會把預設的 pfifo leaf 換成 netem（同為 classless qdisc，但多了 delay / loss / limit 控制能力）：

```
HTB (root)  →  class 1:1  →  netem (limit=Y)     ← 取代 pfifo
```

理解這個三層結構後，接下來的 Phase 0 / 1 / 2 分析就是在追蹤：**drop 到底發生在哪一層、計數器能不能看到。**

以下分三個階段說明我們遇到的問題和演進。

---

## 3. Phase 0：原始狀態 — 只有 HTB，沒有 netem

### 3.1 Mininet `addLink(bw=X)` 做了什麼

當只設定 `bw` 而不設 `max_queue_size` 時，Mininet 的 `TCIntf.config()` 會建立：

```
qdisc htb (root)  ← rate = X Mbps
 └── class 1:1 (rate = X Mbps, ceil = X Mbps)
      └── pfifo (default child)  ← 隱含的 FIFO buffer
```

這裡的 **pfifo** 是 HTB 內部自動建立的 default leaf qdisc。它的 queue 大小由 kernel 決定（通常 = interface 的 `txqueuelen`，預設 1000 packets）。

### 3.2 Drop 發生在哪裡？

當流量超過 `bw` 限制時：

1. 封包進入 HTB root → 被分類到 class 1:1 → enqueue 到 pfifo
2. HTB 以 `rate` 的速度 dequeue → 限制實際吞吐量
3. 如果 pfifo 滿了 → **pfifo 丟棄封包**

### 3.3 為什麼 tc stats 看不到 drop

`tc -s qdisc show dev <NIC>` 的輸出格式如下（數字為示意，非實測值）：

```
qdisc htb 5: root refcnt 25 r2q 10 default 0x1 direct_packets_stat 0 direct_qlen 1000
 Sent 123456 bytes 100 pkt (dropped 0, overlimits 50 requeues 0)
```

第一行是 qdisc 設定（類型 = HTB、handle = `5:`、root、所有流量走 default class `5:1`），其餘為 kernel 內部參數，可忽略。

第二行是統計，各欄位意義：

| 欄位 | 意義 |
|---|---|
| `Sent … bytes … pkt` | 成功從此 qdisc dequeue 送出的封包 |
| `dropped` | 此 qdisc 層級的丟棄計數 — **★問題所在：不含 leaf（pfifo）的 drop** |
| `overlimits` | 封包想 dequeue 但 class 超過 rate limit，暫緩排隊等下次機會（不是 drop） |
| `requeues` | 已 dequeue 但底層 driver 忙碌而塞回的次數 |

**關鍵：`tc -s` 只顯示 HTB root 自身的 stats，不顯示隱含的 pfifo child。** pfifo 的 drop 有時會反映到 HTB 的 `dropped` 計數器，有時不會（取決於 kernel 版本和 drop 的路徑）。

以下是在 1Mbps bottleneck link 上灌 5Mbps UDP 15 秒的**實測結果**（同樣透過 `tc -s qdisc show` 讀取）：

| 指標 | 值 | 來源 |
|---|---|---|
| 預期送出（5Mbps × 15s） | ~6,250 pkt | 手算 |
| tc htb `Sent` | 1,338 pkt | `tc -s` 第二行的 `Sent` |
| tc htb `dropped` | **0** | `tc -s` 第二行的 `dropped` |

約 4,900 個封包消失，但 tc stats 顯示 dropped = 0。這些封包被 pfifo 丟棄了，但計數器沒有正確反映。

### 3.4 Ryu PortStatsReply 也看不到

要理解這點，需先釐清 Mininet 中**兩套獨立的計數器體系**：

```
封包進入 tc qdisc chain (HTB → pfifo)
  │
  ├── 在 pfifo 內被丟棄
  │     → tc "dropped" 可能計、可能不計（上述 §3.3 的問題）
  │     → 封包從未離開 tc → interface tx_packets 完全不計
  │
  └── 成功通過 tc → dev_hard_start_xmit() → 實際發送上線
        → /proc/net/dev 的 tx_packets += 1
        → OVS ofproto 讀取同一個計數器
        → Ryu 發 OpenFlow PortStatsRequest → OVS 以 PortStatsReply 回報此值
```

| 計數器 | 來源 | 計什麼 | 能看到 tc 內部 drop 嗎 |
|---|---|---|---|
| tc `Sent` / `dropped` | `tc -s qdisc show`（Linux shell） | qdisc 內部統計 | `dropped` 有時不準（§3.3） |
| **tx_packets** | `/proc/net/dev` = OVS ofproto = **Ryu PortStatsReply** | 成功離開 qdisc chain、實際上線的封包 | **不能** — 被丟的封包從未被計入 |

> 在絕大多數情況下兩者相等；理論上若 driver 拒收、或 packet 在 qdisc 和 driver 之間被丟（很少見），Sent > tx_packets。Mininet + veth 基本沒有這種 gap。

Ryu 拿到的 `tx_packets` 只計成功離開 qdisc chain、實際上線的封包。因此用 `tx_packets(src) - rx_packets(dst)` 計算 loss，結果永遠 ≈ 0：

```python
# 原本 manager.py 的 loss 公式
loss = (tx_pkts_src - rx_pkts_dst) / tx_pkts_src
# tx_pkts_src 只計算通過 tc 的封包（被 tc 丟掉的不算）
# rx_pkts_dst 是對端 port 收到的（veth pair 不丟封包）
# 兩者幾乎相等 → loss ≈ 0
```

**歷屆實驗觀察到的非零值（0.x%~2%）純粹是 Ryu 對不同 switch 依序 polling 的時間差造成的計數器不同步。**

**結論：在 Phase 0 狀態下，tc stats 和 Ryu PortStatsReply 都無法觀測到真實的 packet loss — 前者計數器不準，後者根本看不到 tc 內部的 drop。**

---

## 4. Phase 1：加入 netem — HTB + netem 兩層架構

Phase 0 的問題是 pfifo 的 drop 計數器不可靠（§3.3）。netem 作為另一種 classless qdisc，有自己獨立的 `dropped` 累積計數器，且可透過 `tc -s qdisc show` 精確讀取。因此我們嘗試用 netem 取代 pfifo 來做 leaf — 在 Mininet 中只需加上 `max_queue_size` 參數即可。

### 4.1 `addLink(bw=X, max_queue_size=Y)` 做了什麼

加上 `max_queue_size` 後，Mininet 建立的 qdisc 變成：

```
qdisc htb 5: root                     ← rate = X Mbps
 └── class 5:1
      └── qdisc netem 10: parent 5:1   ← limit = Y packets
          （取代了原本的 pfifo）
```

netem（Network Emulator）是 Linux kernel 的網路模擬工具，支持 delay、loss、reorder、queue limit 等功能。當 `max_queue_size=Y` 時，Mininet 建立 netem 並設定 `limit=Y`。

### 4.2 預期行為

```
封包進入 HTB → 分類到 class 5:1 → enqueue 到 netem
                                    ↓
                               netem 檢查 backlog < limit?
                                ├── 是 → 放入 queue
                                └── 否 → 丟棄（dropped +1）
                                    ↓
                            HTB dequeue → 以 rate 速度送出
```

netem 的 `dropped` 計數器是精確的累積值，可以通過 `tc -s qdisc show` 讀取。我們基於此建立了 `simple_tc_loss.py` 模組。

### 4.3 實測結果：loss 還是 0

設定 `max_queue_size=1000`，在 GEANT 拓撲上跑 3x traffic scale TM03（LP MLU(directed)=1.2，明確 overloaded）：

| Link (13→14) | 值 |
|---|---|
| bw | 1.55 Mbps |
| free_bw | ≈ 0 |
| delay | ~2 秒 |
| queue_pkts (backlog) | ~275（穩定） |
| netem dropped | **0** |

Queue 很明顯有在堆積（275 packets），但 netem 的 dropped 計數器就是 0。更奇怪的是 queue 穩定在 275，遠低於 limit=1000，永遠不會滿。

---

## 5. 深入分析：HTB 為什麼吞掉了 drop

### 5.1 控制實驗設計

為了隔離問題，我們建立了最簡單的測試環境：

```
h1 --(100M)-- s1 --(1Mbps, queue=Q)-- s2 --(100M)-- h2
Traffic: h1 → h2 UDP 5Mbps（5x overload），持續 30 秒
iperf3 參數：-u -b 5M -t 30 -l 1460（與 GEANT 實驗一致）
```

> iperf3 UDP payload = 1460 bytes → L3 skb = IP(20) + UDP(8) + 1460 = **1488 bytes**
> （tc 計量單位，詳見 [`sim_fluid_queue_calibration.md`](sim_fluid_queue_calibration.md) §參數設定）
>
> 理論總封包數 ≈ 5Mbps × 30s / 8 / 1460 ≈ **12,850 pkt**（以 payload 為 iperf3 `-b` 計量基準）
>
> 實測腳本：[`diagnostics/test_queue_fill.py`](../diagnostics/test_queue_fill.py)

測試兩種 queue size：Q=20 和 Q=1000。

### 5.2 實驗結果

#### queue=20

```
  time      sent   dropped   backlog     drop%
    2s       257       994        20     79.5%
    4s       424      1687        20     79.9%
    ...
   28s      2427      9996        20     80.5%
   30s      2538     10336         0     80.3%

  SUMMARY:
    Total arrived at netem:  12,874
    Total sent (through):    2,538
    netem dropped:           10,336    ← 大量 drop！
    Loss rate:               80.3%     ← 吻合理論值 (5-1)/5 = 80% ✓
    Throughput:              3.79 MB ≈ 1 Mbps ✓
```

#### queue=1000

```
  time      sent   dropped   backlog     drop%
    2s       259         0        47      0.0%
    4s       426         0        48      0.0%
    ...
   28s      2435         0        49      0.0%
   30s      2576         0         0      0.0%

  SUMMARY:
    Total arrived at netem:  2,576     ← 只有 queue=20 的 1/5！
    Total sent (through):    2,576
    netem dropped:           0         ← 完全沒有 drop
    Loss rate:               0%
    Throughput:              3.83 MB ≈ 1 Mbps ✓
```

### 5.3 關鍵對比

| 指標 | queue=20 | queue=1000 |
|---|---|---|
| 到達 netem 的封包總數 | **12,874** | **2,576** |
| netem dropped | 10,336 | 0 |
| netem backlog（穩態） | 20（滿載） | ~47（穩定） |
| HTB overlimits | 2,522 | 2,561 |
| 實際吞吐量（bytes） | 3.79 MB | 3.83 MB |
| iface tx_dropped | 0 | 0 |
| Loss rate | 80.3% | 0% |

**兩組的 byte-level 吞吐量完全一致**（都正確限在 ~1Mbps），但到達 netem 的封包數差了 5 倍。

### 5.4 根本原因：HTB+netem 兩層架構下的 upstream 流控

問題出在 HTB 和 netem 的 **兩層架構互動**。先理清這兩層的職責：

```
                   ┌────────────────────────────────────────┐
    OVS 想送入      │           tc qdisc on s1-eth2          │
  ─────────────►   │                                        │
 dev_queue_xmit()  │   HTB (root)            netem (leaf)   │ ──► 出 NIC
                   │   classifier           buffer, limit=Q │   以 1Mbps
                   │   token bucket                         │   送出
                   │   ＝控制 dequeue 速率                   │
                   └────────────────────────────────────────┘
   enqueue 方向：OVS → HTB 分類 → 放進 netem buffer
   dequeue 方向：netem 出封包 → HTB 以 token bucket 放行 → NIC
```

兩點要澄清：

- **HTB 本身沒有 queue**（本配置下 direct_qlen=1000 僅給未分類的直通封包，我們的流量都走 class 1:1）；所有排隊都在 netem buffer 裡。
- **HTB 的 rate=1Mbps 只限 dequeue**，不限 enqueue。理論上 OVS 想塞多少就能塞多少進 netem，直到 netem backlog == limit 才會在 enqueue 時 tail-drop（計入 netem dropped）。

依此推論，queue=1000 應該在 ~3 秒內填滿 netem 並開始 drop。但實測 **backlog 只到 ~47，drop=0**。這代表 **OVS → s1-eth2 的 arrival rate 實際上被壓到 ≈1Mbps**，壓力點不在 netem 的 tail-drop，而在更上游。

**兩組行為差異的觀測性總結：**

| queue | netem backlog | netem dropped | 到達 netem | 消失在上游 |
|---|---|---|---|---|
| 20   | 20（滿） | 10,336 | 12,874 | ~0 |
| 1000 | ~47     | 0      | 2,576  | ~10,300 |

- queue=20 時 netem 一進就滿，drop 是 O(1) 快速路徑 → qdisc 不呈現「忙」狀態 → 上游不減速，全部 ~12,800 pkt 都到得了 netem。
- queue=1000 時 netem 不會瞬間滿 → qdisc 長時間處於「HTB 沒 token、等 watchdog」的狀態 → 上游某條路徑偵測到這個狀態並壓制送入速率 → netem 永遠餓不到 limit。

**關鍵機制：HTB 的 rate-limit 是 token bucket + watchdog 實作的。** 當 token 用完，HTB 會把 qdisc 明確標記為 **throttled state**（`__QDISC_STATE_THROTTLED`），由 watchdog timer 在下一個 token 可用時再喚醒。只要 qdisc 處於 throttled，就有 kernel 層的 flow control 沿著 `dev_queue_xmit()` → veth → OVS datapath 路徑生效，把封包擋在 tc 之外（精確傳播路徑未實證定位，可能是 veth tx 排隊、OVS per-port backlog、或 softirq 排程）。重點是：這個壓制是 **HTB 的 throttled state 啟動的**，因此只要換成不會把 qdisc 推進 throttled 的 rate-limit 機制，問題就消失（後面 §6 會說明）。

對 loss 量測的結論：**drop 發生在 tc 看不到的層**，`tc -s qdisc show` 的 HTB/netem 兩層 dropped 都是 0，`/proc/net/dev tx_dropped` 也是 0。

### 5.5 那多餘的封包去哪了？

實測全量約 12,874 packets（§5.2 queue=20 的基準，接近理論值 ~12,850），但 queue=1000 時 netem 只看到 2,576。剩下的 ~10,300 packets 被消耗在 tc 之外：

```
h1 iperf3 sendto()
    │
    ▼
h1 kernel UDP socket buffer
    │
    ▼
h1-eth0 → veth → s1-eth1            (100Mbps, 不是瓶頸)
    │
    ▼
OVS kernel datapath                  ← ★ 封包在某個位置消失 ★
    │                                   （OVS per-port backlog / veth tx /
    │                                    softirq 排程 — 未進一步實證定位）
    ▼
s1-eth2 的 tc qdisc (HTB+netem)     ← 我們的 tc -s 只看到這一層
```

**這些 drop 發生在 tc qdisc 之外**（OVS kernel datapath 或 veth/netdev 層），不會被 `tc -s qdisc show` 記錄，也不會被 `/proc/net/dev` 的 `tx_dropped` 記錄（實測 `iface tx_dropped = 0`）。

這就是為什麼我們在 Phase 1 中看到 loss = 0 — **drop 確實發生了，但發生在我們能觀測到的位置之外。**

---

## 6. Phase 2：根治方案 — netem-only

### 6.1 設計理念

§5.4 的結論：HTB 用 token bucket + watchdog 實作 rate-limit，沒 token 時會把 qdisc 推進 **throttled state**，這個狀態會沿 kernel 層壓制上游送入速率，讓 netem 永遠收不到超過 ~1Mbps 的流量。drop 因此發生在 tc 看不到的層、netem 自己的 `dropped` 計數器恆為 0。

關鍵是 **netem 的 `rate` 參數用完全不同的機制實作**：它給每個封包算 virtual departure time，dequeue 在時間未到時回傳 NULL，**但不把 qdisc 標記為 throttled**。因此 netem-only 架構下沒有 throttled state 可供上游 flow control 偵測，封包得以全速 enqueue 進 buffer。

根治方案：**拆掉 HTB，用 netem 一層做 rate + buffer + drop**，拿掉「會把 qdisc 標記為 throttled 的那一層」，所有 drop 就落在可觀測的 netem `dropped` 計數器內。

netem 本身支持 `rate` 參數，可以模擬 link bandwidth（對每個封包根據大小計算 virtual dequeue delay）：

```bash
# Phase 1（HTB + netem）：
tc qdisc add dev s1-eth2 root htb default 1
tc class add dev s1-eth2 parent 1: classid 1:1 htb rate 1mbit
tc qdisc add dev s1-eth2 parent 1:1 netem limit 1000

# Phase 2（netem-only）：
tc qdisc add dev s1-eth2 root netem rate 1mbit limit 1000
```

架構對比：

```
Phase 1:   HTB (rate limit) → netem (buffer)     ← 兩層，drop 被隱藏
Phase 2:   netem (rate + buffer + drop)           ← 一層，所有 drop 可見
```

### 6.2 為什麼 netem-only 的 drop 能被看見

在 netem-only 架構下：

1. netem 是 root qdisc，唯一一層
2. 所有從 OVS 來的封包直接 enqueue 到 netem
3. netem 的 `rate` 參數只控制 **dequeue 速度**，不影響 enqueue
4. 封包以 OVS 推送的速率進入 netem buffer
5. 當 buffer 滿了（backlog = limit）→ **netem 直接 tail-drop，`dropped` += 1**
6. Tail-drop 是 O(1) 快速路徑（和 §5.4 queue=20 的情況同樣），qdisc 不呈現「忙」→ 上游不被壓制 → OVS 持續全速推送

實證結果（§6.3）：netem 看到的到達量 12,117 / 理論 ~12,850 ≈ 94%，drop 計數器與實際 loss rate 吻合。

### 6.3 驗證實驗

同一個測試環境（1Mbps bottleneck、5Mbps UDP、queue=1000），比較兩種模式：

同一個測試腳本 `diagnostics/test_queue_fill.py`（iperf3 UDP, 5Mbps send, `-l 1460`, 30s），queue=1000：

| 指標 | HTB+netem (Phase 1) | netem-only (Phase 2) |
|---|---|---|
| 到達 netem | 2,576 | **12,117** |
| netem dropped | 0 | **9,355** |
| Loss rate | 0% | **77.2%** |
| Backlog | 穩定在 ~47（遠低於 limit） | **1000（滿載）** |
| Throughput | ≈ 1 Mbps | ≈ 1 Mbps |

- 兩種模式的 **byte-level 吞吐量完全一致**（都正確限在 1Mbps）
- netem-only 的 loss rate 77.2% 接近理論值 (5−1)/5 = 80%（差距來自 iperf3 首秒 ramp-up — pacing scheduler 尚未收斂到 5Mbps；與尾端 drain — t=30s 停送時 netem 內還積著 ~7s 的 buffer 未被 summary window 計入）
- netem-only 模式下 netem 看到的到達量 12,117 接近理論總量 ~12,850，而 HTB+netem 只看到 2,576 — 差額 ~9,500 packets 被上游流控擋在 OVS/kernel 層（位置未實證定位，詳見 §5.4–5.5）

**netem-only 方案驗證成功：同樣的 queue=1000，只是去掉 HTB，drop 就完全可見。**

### 6.4 實作方式

在 `env_loader.py` 中，`net.start()` 之後，對所有 switch-switch link 執行 tc 替換：

```python
# 刪除 Mininet 自動建立的 HTB+netem
tc qdisc del dev {intf} root
# 建立 netem-only
tc qdisc add dev {intf} root netem rate {bw}mbit limit {queue_size}
```

`simple_tc_loss.py` **不需要修改** — 它已經在讀 netem 的 stats，只是之前 netem dropped 永遠是 0 而已。

---

## 7. 與真實 SDN ASIC 交換機的對照


### 7.1 Per-port queue 大小與 link capacity 的關係(queue_size設計)

現代 ASIC 交換機多半採 **shared memory (MMU) 架構**，per-port buffer 從共用池動態分配，設計上 **per-port 配額通常和該 port 的 line rate 無關**（或僅弱相關）— 同一顆 ASIC 上 10GbE / 100GbE / 400GbE 埠往往共用同一個 pool，分配策略看 QoS、班級、壅塞程度而非線速。

這說明**業界常見作法就是「所有 port 配固定（或同量級）的 buffer」**，不是按 link BW 等比例縮放。DRL-OR-S 與我們的 Mininet 模擬都延用此作法（§8.1 詳列）：

| 設定 | 所有 port 的 queue size |
|---|---|
| 我們（GEANT /100 scaling） | 固定 Q=1000 packets |
| DRL-OR-S（/1000 scaling） | 固定 Q=100 packets |

兩者的共通點：**queue size 和 link BW 解耦**。差異只在 scaling factor（/100 vs /1000）導致 packet 數不同，這個差異會直接影響 max queuing delay（§8.2 會算給出各 link 的合理度）。

---

## 8. 我們的 Buffer Sizing 分析

### 8.1 Bandwidth scaling 的影響

我們的 GEANT 拓撲使用 **/100 BW scaling**：

| 真實 link | 真實 BW | Mininet BW | 說明 |
|---|---|---|---|
| OC-192/STM-64 | 10 Gbps | 100 Mbps | 骨幹 link |
| OC-48/STM-16 | 2.5 Gbps | 25 Mbps | 中等 link |
| STM-1 | 155 Mbps | 1.55 Mbps | 窄 link |

DRL-OR-S 使用 /1000 scaling（`float(c) / 1000`）。

### 8.2 Max queuing delay 計算

```
max_queuing_delay = queue_pkts × pkt_size × 8 / bandwidth
```

| 設定 | Link BW | queue | Max delay | 備註 |
|---|---|---|---|---|
| **我們（/100, Q=1000）** | 1.55 Mbps | 1000 | **7.74 s** | 極度不合理 |
| 我們（/100, Q=1000） | 25 Mbps | 1000 | 0.48 s | 偏大 |
| 我們（/100, Q=1000） | 100 Mbps | 1000 | 0.12 s | 合理 |
| **DRL-OR-S（/1000, Q=100）** | 2.488 Mbps | 100 | **0.48 s** | 偏大 |
| DRL-OR-S（/1000, Q=100） | 9.953 Mbps | 100 | 0.12 s | 合理 |
| 真實 GEANT（WAN router Q≈100–1000） | 155 Mbps | ~100–1000 | ~7.7–77 ms | GEANT 官方未公開 per-port queue size；WAN provider router 典型值為 100–1000 packets 等級 |
| 真實 GEANT（WAN router Q≈100–1000） | 10 Gbps | ~100–1000 | ~0.12–1.2 ms | 同上 |

**問題：固定 queue size 配合被 /100 縮放的 link BW，會讓 max queuing delay 被放大 ~100×。**

- 我們的 1.55 Mbps link 配 Q=1000 → 7.74 秒（荒謬；真實 155 Mbps 配同等 Q 只有 77 ms）
- DRL-OR-S 的最慢 link 配 Q=100 → 0.48 秒（同樣受 /1000 BW scaling 影響，程度較輕）

### 8.3 未來若要貼近現實(未實作)：queue size 跟著 BW 同步縮放

真實 ASIC 的慣例是 per-port 固定 buffer（§7.2），max queuing delay 因 BW 而異。模擬時要保留這個「時間尺度」，queue size 必須和 BW 做**相同比例**的縮放：

```
真實：155 Mbps × Q=300 → max delay ≈ 23 ms
我們（/100 scaling）：1.55 Mbps × Q=3 → max delay ≈ 23 ms ✓
```

換句話說，**BW scale 了 /100，Q 也要跟著 /100**（例如目前的 Q=1000 應降為 Q=10 量級）。這樣模擬出的 queuing delay 才和真實網路同量級，RL state 讀到的 delay 分佈也才具物理意義。

> 注意：這是 buffer sizing 的設計問題，和 loss 測量方案（netem-only）是正交的。
> netem-only 方案在任何 queue size 下都能正確測量 loss。

---

## 9. DRL-OR-S 的對照

DRL-OR-S testbed 的完整設定：

```python
# DRL-OR-S testbed.py
self.addLink(sw1, sw2,
    bw = capacity_Kbps / 1000,    # /1000 scaling
    delay = '5ms',                 # 固定 propagation delay
    loss = loss_pct,               # netem random loss（%）
    max_queue_size = 100)          # netem limit
```

注意：DRL-OR-S 也設了 `max_queue_size`，所以它的 tc 也是 HTB+netem。但因為 queue 只有 100，在 /1000 scaling 下比較容易填滿（最慢的 link max delay 也才 0.48s）。如果流量確實超載，netem 有機會填滿並 drop。

不過 DRL-OR-S 並未使用 `tc -s` 來讀 loss — 它使用 **application-layer UDP 探針**（在 server.py 中用 sequence number 計算收到/遺失的封包數）。這完全繞過了 tc stats 的問題。

我們的 tc-based 方案在改為 netem-only 後，也能達到同樣效果，且不需要額外的 application-layer 探針。

### 我們和 DRL-OR-S 還有一個差異：propagation delay

DRL-OR-S 設了 `delay='5ms'`（所有 link 統一 5ms propagation delay）。我們目前沒有設定 delay（propagation delay = 0）。

---

## 10. 總結：三個階段的演進

```
Phase 0: addLink(bw=X)
    tc: HTB → (implicit pfifo)
    Drop: 在 pfifo 中，tc -s 不可見
    Loss 測量: ✗ 不可能

Phase 1: addLink(bw=X, max_queue_size=Y)
    tc: HTB → netem(limit=Y)
    Drop: 被 OVS/kernel 層上游流控擋住，netem 永遠不滿
    Loss 測量: ✗ netem dropped = 0

Phase 2: netem-only (rate=X, limit=Y)    ← 根治方案
    tc: netem(rate=X, limit=Y)
    Drop: 在 netem 中，tc -s 完全可見
    Loss 測量: ✓ 精確
    與真實 ASIC: 最接近（單層 output buffer）
```

### 未來可選優化

1. 參考 DRL-OR-S 的設計，為每條 link 加上固定的 propagation delay（從 `bw_r.txt` 讀 latency，透過 netem 的 `delay` 參數注入），讓模擬的 end-to-end delay 更貼近真實網路。否則 veth pair 本身的傳播延遲近乎 0，RL agent 在 state 裡看到的 delay 幾乎全由 queuing 主導，缺少真實 WAN 的 propagation baseline。
2. 讓 queue size 跟著 capacity 同步縮放以貼近現實