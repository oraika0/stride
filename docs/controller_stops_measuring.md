# controller 停止量測，而且沒有任何東西發現

一次 32-node 的 real Mininet 實驗跑到第 591 步，reward 曲線還在動，checkpoint 也照常
存下來，而它看到的最新一筆鏈路量測已經是 85 分鐘前的了。訓練迴圈從頭到尾沒有發現，
因為它裡面沒有任何一處會去看。

這篇記錄兩個症狀相同但成因不同的失敗，都是在同一台機器上、幾個小時之內發現的。一個
已經理解、重現、修好。另一個只量到現象，還沒有解釋，寫下來是為了讓下一個遇到的人認得
出來，而不是重新發現一次。

## 症狀，以及為什麼平常看的訊號都抓不到

卡住的實驗從任何一個平常會看的角度看都很健康。步數在前進，checkpoint 有寫出來，reward
曲線也還在動 —— reward 跟著動作走，所以在它背後的鏈路量測早就不再變化之後，它仍然會
繼續對策略做出反應。有一次凍結的實驗在 40 個凍結的步數上產生了 40 個不同的 reward 值。

**MLU 釘在 1.0 是線索，不是證據。** 一個始終沒學會疏解飽和的策略會畫出一模一樣的平線。

決定性的判斷是量測到底還有沒有在寫。

```bash
date; stat -c %y results/<alg>/net_info_directed.csv
```

跑完的實驗還會歸檔一份 `measurement.txt`，裡面有 cycle 數，以及最後一筆量測是在結束前
多久落下的。只要超過一個監測週期，就代表 agent 當時讀的是一份不會再更新的檔案。

## 失敗 A —— 讀到寫到一半的檔案，monitor greenthread 因此死掉

**已證實、可重現、已修復。**

controller 與 DRL agent 是兩個獨立的行程，靠 `results/<alg>/` 底下四個檔案溝通。

| 檔案 | 方向 | 大小 |
| --- | --- | --- |
| `drl_paths.json` | agent → controller | 約 88 KB，每步重寫 |
| `paths_metrics.json` | controller → agent | 約 1.3 MB，每個 cycle 重寫 |
| `net_info.csv` | controller → agent | 約 2 KB |
| `net_info_directed.csv` | controller → agent | 約 6 KB |

四個都是用 `open(path, "w")` 寫的。這個寫法會**先立刻把檔案截斷**，然後以 8 KB 為單位
分批 flush。在整個寫入期間，磁碟上的檔案是新內容的一段前綴，而讀的那一方無法分辨它跟
一份完整的檔案有什麼不同。

controller 就是死在這件事上。

```
File "utils/simple_monitor.py", line 394, in get_dRL_paths
    paths_dict = json.load(json_file)
json.decoder.JSONDecodeError: Expecting value: line 815 column 7 (char 8192)

During handling of the above exception, another exception occurred:
  File "utils/simple_monitor.py", line 109, in monitor
  File "utils/simple_monitor.py", line 403, in get_dRL_paths
json.decoder.JSONDecodeError: Expecting ',' delimiter: line 4861 column 6 (char 49129)
```

`char 8192` 剛好是兩次 buffer flush 的位置 —— 讀的那一方接在兩個區塊中間讀到了。

`get_dRL_paths` 本來就有在 `time.sleep(0.35)` 之後重試一次，但**重試本身沒有再包一層
保護**，所以第二次讀又落在寫入過程中的時候，例外就直接逃進 monitor greenthread。
`ryu.lib.hub` 會把 traceback 印出來，然後讓那個 greenthread 結束。controller 其他部分
完全不受影響 —— event loop、receive loop、延遲偵測、拓撲 worker 全都還活著，行程還在
吃 CPU，只是監測 cycle 再也沒有跑過。指標凍結在 cycle 85，而 agent 就這樣一路訓練到
有人去把它關掉為止。

那個重試，還有它旁邊被註解掉的
`except ValueError as e: #error exception when trying to read the json and is still been updated`，
等於明白說了這個 race 當初就知道，只是繞過去而不是解決掉。

這個時間窗有多寬。讓一個行程持續重寫檔案、另一個行程持續讀，各跑六秒。

```
plain open('w')   152,782 次讀取   152,755 次讀到殘缺   (100.0%)
atomic replace     27,391 次讀取         0 次讀到殘缺   (  0.0%)
```

### 修法

`utils/atomic_io.py` 先寫進目標檔案**同一個目錄**底下的暫存檔，再用 `os.replace()` 蓋
過去。這個 rename 在同一個檔案系統內是原子的，所以讀的那一方看到的要嘛是完整的舊版本，
要嘛是完整的新版本。`utils/manager.py` 與 `loader/train_loader.py` 裡全部九個寫入點都
改走這條路。

另外兩個後續修改，因為**把成因修掉，跟讓失敗變得可以存活，是兩件事**。

- `get_dRL_paths` 不再往外丟例外。萬一還是讀失敗，它記錄下來然後沿用上一個 cycle 的
  路由。重複用一個 cycle 的路徑，比從此不再量測是小得多的錯誤。
- monitor 的迴圈主體搬進 `_monitor_cycle`，由 `monitor` 在外面接例外、記錄 traceback
  與連續失敗次數，然後繼續跑。**一個丟出例外的 greenthread 就永久消失了**，而這已經是
  這類失敗第三次被記錄下來 —— `MONITOR_START_DELAY` 上面那段註解記的是更早的一次，
  當時是拓撲還不完整造成的 `KeyError`。

原子替換順便也拿掉了兩處呼叫點原本的 `os.remove()` 再重建的把戲。那是為了繞過前一次
sudo 執行留下的、屬於 root 的 `drl_paths.json` —— 把新檔案 rename 蓋過目標，需要的是
**目錄**的寫入權限，不是檔案的。

## 失敗 B —— receive loop 停了，原因不明

**只量到，沒有解釋。** 同一台機器，早幾個小時，而且**不是**失敗 A —— 這裡 monitor
greenthread 是活的，還在印 cycle 601、602、603，只是每一個 cycle 都跟著一則 port-stats
警告。

```
[monitor] cycle 601: 0/32 switches returned port stats after 3.0s
```

是 0，不是部分回來，而且每個 cycle 都是 0。當時這個行程的樣子。

| 量到的東西 | 數值 | 怎麼量的 |
| --- | --- | --- |
| OpenFlow socket 上沒讀走的資料 | 26.5 MB，每條約 830 KB | `ss -tn state established '( sport = :6653 )'` 的 Recv-Q |
| 消化速率 | 18 秒內 0 B/s | 兩次 Recv-Q 取樣 |
| 在 hub 的 epoll 集合裡的 socket | 34 條裡只有 2 條，而且都是 listener | `/proc/<pid>/fdinfo/<epollfd>` 的 `tfd:` 行 |
| 行程狀態 | `S (sleeping)`，`wchan = ep_poll` | `/proc/<pid>/wchan` |
| controller CPU | 一顆核心的 27 % | `top -bn2 -p <pid>` |
| 機器負載 | 32 核上 1.97 | `/proc/loadavg` |

**eventlet 只有在某個 greenthread 正在等著讀某個 descriptor 的時候，才會去 watch 它。**
32 條 switch 連線全部不在 epoll 集合裡，代表當時沒有任何 greenthread 在讀它們 —— hub
不是忙不過來，而是根本沒有東西登記要被喚醒。與此同時 controller 還在往同一批 socket
**寫**大約 40 KB/s 的 flow modification，而且 `[Flow Installation Ok]` 照樣在印，因為
那只是一個無條件的 `print`，不是確認訊息。

最自然的解釋是 Ryu 的 backpressure。每個 app 背後只有一個 greenthread 與一個有界佇列，
佇列滿了之後投遞就會擋住生產者。

```python
# ryu/base/app_manager.py:160
self.events = hub.Queue(128)
self._events_sem = hub.BoundedSemaphore(self.events.maxsize)

# ryu/base/app_manager.py:301
def _send_event(self, ev, state):
    self._events_sem.acquire()
    self.events.put((ev, state))
```

生產者就是每個 datapath 的 receive loop，所以只要有一個 app 停止消化，它就會擋住每一條
socket，而且是永久的，因為它在等的東西本身也得從沒有人在讀的那些 socket 過來。

**這個解釋沒有被驗證過。** 這次事故發生的時候 dump 工具還不存在，而下一次用 dump 抓到
失敗的時候，每一個佇列都是 `0 / 128` —— 那次結果是失敗 A，是另一回事。所以這個佇列理論
從來沒有真的被觀察到，失敗 B 也可能根本是別的東西。如果它再發生，`kill -USR1` 一次就能
問出答案。

### handler 規則，以及一個因為它自己就該改而改掉的地方

在追失敗 B 的過程中發現 `utils/simple_awareness.py` 的 `get_topology` 是一個**會擋住的
handler**。它呼叫 `get_switch()` 與 `get_link()`，而這兩個不是本地查詢，是對 `Switches`
app 的同步請求。

```python
# ryu/topology/api.py:20
def get_switch(app, dpid=None):
    rep = app.send_request(event.EventSwitchRequest(dpid))
    return rep.switches

# ryu/base/app_manager.py:265
def send_request(self, req):
    req.sync = True
    req.reply_q = hub.Queue()
    self.send_event(req.dst, req)
    return req.reply_q.get()          # 這裡會擋住
```

Ryu 自己的文件禁止這種寫法，在 `doc/source/ryu_app_api.rst` 的
*Threads, events, and event queues* 一節。

> Because the event handler is called in the context of the event processing
> thread, it should be careful when blocking. While an event handler is
> blocked, no further events for the Ryu application will be processed.

也就是 handler 跑在事件處理執行緒的脈絡裡，它被擋住的期間，這個 app 的其他事件全部不會
被處理。

**要精確講上游沒有說的部分。** Ryu 從來沒有把「在 handler 裡面同步呼叫」稱為死鎖。
`deadlock` 這個字在整個原始碼樹裡只出現一次，就在同一份檔案再往下幾行，而且講的是相反
的事情 —— 為什麼回覆要有自己的佇列。

> While such requests uses the same machinary as ordinary events, their replies
> are put on a queue dedicated to the transaction to avoid deadlock.

那個專用的回覆佇列，防的是回覆卡在呼叫者自己的積壓後面。它對「被呼叫的那個 app 本身就
是卡住的那個」完全沒有幫助。把文件裡的 handler 規則跟有界佇列的 backpressure 合起來
推論，是這裡下的結論，不是從上游引用來的警告。

現在這個 handler 全部就只有這樣。

```python
@set_ev_cls(events)
def get_topology(self, ev):
    self._topo_dirty = True
```

實際的重建由 `_topology_worker` 在它自己的 greenthread 上做，每
`setting.TOPO_REBUILD_PERIOD`（0.5 秒）一次。原本的 handler 主體原封不動搬成
`_rebuild_topology`。對 `utils/` 底下每一個 `@set_ev_cls` handler 做了一次 AST 掃描，
十個全部乾淨，`get_topology` 是唯一一個違規的。

**這個修改並沒有修好接下來實際觀察到的那次失敗。** 它是真的違反規則、真的該拿掉，而且
合併重建確實實質減少了跨 app 的請求量 —— 拓撲發現時那一連串 `LinkAdd` / `PortAdd` 事件
以前是每一個都觸發一次完整重建，而每次重建會發出多達上百個請求 —— 但它之後那次實驗死於
失敗 A，而失敗 B 再也沒有出現過，所以無從得知它有沒有解決那件事。

## 為什麼跑論文結果的那台機器兩個都沒遇到

論文裡的每一個結果都來自 PC0（i7-13700K），26 次歸檔的 real Mininet 實驗，其中 24 次是
32-node，大約 78,000 步，全程 `sim_training = False`。歸檔的輸出都檢查過有沒有 MLU 長段
不變、以及量測檔案是不是比實驗本身還舊，結果乾淨。

**PC0 對失敗 A 並沒有免疫**，它跑的是同一份有 race 的寫法。它幾乎確定反覆撞到過讀取
殘缺，然後每次都復原了，因為那一次重試就夠了 —— 第二次讀是 0.35 秒之後，在那台機器上
一個 88 KB 的寫入早就結束了。真的死掉的那次，需要**兩次讀取都**落在寫入過程中，那要求
寫的那一方慢到、或忙到三分之一秒之後還在寫。

失敗 B 這邊的差異就比較不確定，因為機制不明。能講的只有量到的部分 —— 卡住的那台是共用的
Xeon Silver 4110（基礎 2.1 GHz、turbo 3.0，當時有十個登入 session 與其他使用者的工作），
對上 PC0 的 i7-13700K 可以衝到約 5.4 GHz。Ryu 是單執行緒的，所以任何跟佇列深度有關的
東西都跟著單核速度走，而這正好差了大約 2.5 到 3 倍。

注意**不在**這張清單上的是什麼 —— CPU 總量。卡住的那台是 32 核、負載 2。兩個失敗都是在
資源要多少有多少的情況下發生的，所以「換一台大一點的機器跑」對兩者都不是解法。

**32-node 這個拓撲比機器本身更關鍵。** 它產生的 LLDP packet-in 流量遠比 Geant 的 23 個
節點多，共用檔案也大得多。歸檔裡那兩次 Geant 實驗從來沒有接近過。

## 下一次怎麼診斷

**`kill -USR1`。** controller 啟動時就會掛上 `utils/greenthread_dump.py`。

```
kill -USR1 $(pgrep -f 'ryu-manager --observe-link')
```

它會印出每個 app 的佇列深度 —— 停在最大值的那個就是卡住的那個 —— 並且把每一個
greenthread 的堆疊寫進
`artifacts/greenthread_dump_<pid>_<timestamp>.txt`。**在 dump 裡「不見了」的
greenthread，就是死掉的那個**。失敗 A 就是這樣認出來的，114 個堆疊裡面沒有任何一個
`simple_monitor.monitor` 的 frame。

只有佇列表會進終端機。第一次跑這個工具的時候它印了 1,600 行堆疊，把真正解釋失敗原因的
那段 traceback 擠出了 2,000 行的 scrollback，這也是為什麼現在 tmux launcher 會把 pane
的輸出另外寫進 `results/_terminal_logs/<window>_<timestamp>.log`。

**py-spy 診斷不了上面任何一件事**，在出事當下伸手去拿它之前值得先知道。greenthread 不是
OS 執行緒，這個行程從頭到尾只有一條，所以 py-spy 回報的是當下正在跑的那個 greenthread，
也就是卡在 `epoll_wait` 的 hub，而這件事 `/proc/<pid>/wchan` 免費就能告訴你。它沒有
greenlet 支援，而且只要 `kernel.yama.ptrace_scope` 是 1，attach 就需要 root。
