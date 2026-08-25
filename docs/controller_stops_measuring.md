# controller 停止量測

## 症狀

量測停止更新，但訓練繼續跑。錯誤**只會出現在 controller 的 pane**，其餘 pane 什麼都
不會印，訓練迴圈也不會發現，因為它裡面沒有任何一處會去看量測的時間戳。silent fail。

reward 曲線還是會動，因為 reward 跟著動作走。MLU 釘在 1.0 是線索不是證據，一個沒學會
疏解飽和的策略會畫出一樣的平線。

確認的方法只有一個。

```bash
date; stat -c %y results/<alg>/net_info_directed.csv
```

超過一個監測週期，agent 讀的就是不會再更新的檔案。跑完的實驗另外有歸檔的
`measurement.txt`，記錄 cycle 數與最後一筆量測落在結束前多久。

下面兩個失敗症狀相同、成因不同，同一台機器上幾小時內先後出現。A 已修好，B 沒有解釋。

## 失敗 A —— 讀到寫到一半的檔案（已修）

controller 與 agent 是兩個行程，靠 `results/<alg>/` 底下四個檔案溝通，沒有 socket。

| 檔案 | 方向 | 大小 |
| --- | --- | --- |
| `drl_paths.json` | agent → controller | 約 88 KB，每步重寫 |
| `paths_metrics.json` | controller → agent | 約 1.3 MB，每 cycle 重寫 |
| `net_info.csv` | controller → agent | 約 2 KB |
| `net_info_directed.csv` | controller → agent | 約 6 KB |

四個都用 `open(path, "w")` 寫。這個寫法先截斷檔案再以 8 KB 分批 flush，寫入期間磁碟上
就是一段前綴，讀的一方分辨不出來。

```
json.decoder.JSONDecodeError: Expecting value: line 815 column 7 (char 8192)
```

`char 8192` 就是兩次 flush 的位置。`get_dRL_paths` 本來有 `sleep(0.35)` 重試一次，但
**重試沒有再包保護**，第二次也撞上就把例外丟進 monitor greenthread。`ryu.lib.hub` 印完
traceback 就讓它結束，controller 其他部分全部正常，只是監測 cycle 再也不跑。

時間窗有多寬，一邊持續重寫一邊持續讀各六秒。

```
plain open('w')   152,782 次讀取   152,755 次殘缺   (100.0%)
atomic replace     27,391 次讀取         0 次殘缺   (  0.0%)
```

### 修法

`utils/atomic_io.py` 寫進同目錄的暫存檔再 `os.replace()`。同一個檔案系統內 rename 是
原子的，讀到的要嘛完整舊版要嘛完整新版。`utils/manager.py` 與 `loader/train_loader.py`
九個寫入點全部改走這條。

另外兩處，因為修掉成因跟讓失敗可存活是兩件事。

- `get_dRL_paths` 不再往外丟例外，讀失敗就沿用上一個 cycle 的路由。重複一次路徑遠小於
  從此不再量測。
- monitor 迴圈主體搬進 `_monitor_cycle`，外層接例外、記連續失敗次數、繼續跑。
  **丟出例外的 greenthread 就永久消失**，而這是這類失敗第三次被記錄。

## 失敗 B —— receive loop 停了（未解）

monitor greenthread 還活著，還在印 cycle，但每個 cycle 都是

```
[monitor] cycle 601: 0/32 switches returned port stats after 3.0s
```

是 0 不是部分回來。當時量到的。

| 量到的東西 | 數值 |
| --- | --- |
| OpenFlow socket 沒讀走的資料 | 26.5 MB，每條約 830 KB |
| 消化速率 | 18 秒內 0 B/s |
| 在 epoll 集合裡的 socket | 34 條裡只有 2 條，都是 listener |
| 行程狀態 | `S (sleeping)`，`wchan = ep_poll` |
| CPU / 負載 | 一核 27 % / 32 核上 1.97 |

eventlet 只在有 greenthread 等著讀時才 watch 那個 descriptor。32 條連線全不在 epoll
集合裡，代表沒有人在讀 —— hub 不是忙不過來，是沒有東西登記要被喚醒。同時 controller
還在往那些 socket 寫 flow modification，`[Flow Installation Ok]` 照印，因為那只是一個
無條件的 `print`。

最自然的解釋是 Ryu 的 backpressure。每個 app 只有一個 greenthread 配一個
`hub.Queue(128)`，滿了就擋住生產者，而生產者正是各 datapath 的 receive loop，所以一個
app 停止消化會永久擋住每一條 socket。

**這個解釋沒被驗證過。** 事發時 dump 工具還不存在，下次用 dump 抓到時每個佇列都是
`0 / 128`，而那次是失敗 A。B 沒有再出現過。再遇到就 `kill -USR1`。

## 附帶修掉的 handler 違規

`utils/simple_awareness.py` 的 `get_topology` 會擋住 —— 它呼叫 `get_switch()` 與
`get_link()`，那是對 `Switches` app 的**同步請求**（`send_request` 最後是
`reply_q.get()`）。Ryu 文件明文禁止在 handler 裡擋住。

> While an event handler is blocked, no further events for the Ryu application
> will be processed.

要精確講，上游從沒把「handler 裡同步呼叫」稱為死鎖，`deadlock` 在整棵樹只出現一次而且
講的是相反的事。把 handler 規則跟有界佇列合起來推論，是這裡下的結論，不是引用來的。

現在 handler 只剩一行 `self._topo_dirty = True`，重建交給 `_topology_worker` 每 0.5 秒
做一次。`utils/` 底下十個 handler 掃過，只有這一個違規。

**它沒有修好實際觀察到的那次失敗**，之後那次死於 A，而 B 沒再出現。

## 為什麼 PC0 沒遇到

論文結果全來自 PC0（i7-13700K），26 次歸檔的 real 實驗約 78,000 步，檢查過沒有 MLU 長段
不變、也沒有量測檔比實驗還舊。

PC0 對 A 並不免疫，它跑的是同一份有 race 的寫法，只是那一次重試就夠了 —— 0.35 秒後
88 KB 的寫入早就結束。死掉的那次需要**兩次讀取都**撞上寫入。

B 這邊機制不明，只能講量到的差異。卡住的是共用的 Xeon Silver 4110（2.1 GHz 基礎，十個
登入 session），PC0 是 i7-13700K 衝到約 5.4 GHz。Ryu 單執行緒，跟佇列有關的東西都跟著
單核速度走。

**不在這張清單上的是 CPU 總量**，卡住那台是 32 核負載 2。兩個失敗都發生在資源充裕時，
所以「換大機器」對兩者都不是解法。

32-node 這個拓撲比機器更關鍵，LLDP packet-in 與共用檔案都遠大於 Geant 的 23 個節點，
歸檔裡兩次 Geant 實驗從沒接近過。

## 診斷

```bash
kill -USR1 $(pgrep -f 'ryu-manager --observe-link')
```

印出每個 app 的佇列深度（停在最大值的就是卡住的），並把所有 greenthread 堆疊寫進
`artifacts/greenthread_dump_<pid>_<ts>.txt`。**在 dump 裡不見的 greenthread 就是死掉
的** —— 失敗 A 就是這樣認出來的，114 個堆疊裡沒有 `simple_monitor.monitor`。

只有佇列表會進終端機。第一次跑時印了 1,600 行堆疊，把解釋原因的 traceback 擠出
scrollback，這也是現在 pane 輸出會另外寫進 `results/_terminal_logs/` 的原因。

**py-spy 診斷不了這些。** greenthread 不是 OS 執行緒，行程只有一條，py-spy 只會回報卡在
`epoll_wait` 的 hub，那件事 `/proc/<pid>/wchan` 免費就有。它沒有 greenlet 支援，而且只要
`kernel.yama.ptrace_scope` 是 1 就需要 root 才能 attach。`kill -USR1` 不用。
