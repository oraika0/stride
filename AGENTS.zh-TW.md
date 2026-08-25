# AGENTS.zh-TW.md

給 coding agent 的單一指示檔的**中文對照翻譯**。實際被工具讀取的是英文版
[`AGENTS.md`](AGENTS.md)，刻意不做任何工具專屬的版本，一份檔案給所有 agent。這份是給人
看的，改動請以英文版為主，兩邊要一起改。

## 基本規則

- 這個 repo 是**程式與實驗優先**。優先做程式修改、實驗、除錯與 runtime 調查，而不是寫
  文件。
- `results/` 跟 `dataset/` 存的是實驗資料與輸入。**沒有被要求就不要改動或刪除**裡面的
  任何東西。
- `A-Traffic-Engineering-.../` 是 vendored 的上游程式碼。只有 `SAC_PL_KP`（gym 環境）跟
  `Enero_datasets`（流量資料）有在用。不要編輯它。
- `paper/` 從 `results/<alg>/runs/<run>/test/` 重新產生論文的圖與表。改動那裡的路徑可能
  安靜地產出一張空圖而不是報錯——任何搬移之後都要確認 glob 還解析得到東西。
- 這個 repo **沒有測試套件**。`diagnostics/` 放的是獨立的方法論實驗，不是 unit test，
  也不會自動執行任何東西。

## 這個專案在做什麼

STRIDE 為每個 OD pair 從一組凍結的 K=20 候選路徑中挑一條。所有 pair 由一個 discrete-
diffusion decoder **共同決定**（從全遮罩狀態出發，經過 M 個 denoise step），並在真實的
Mininet + Ryu 測試床上用 actor–critic RL 訓練。

**`algs/stride.py` 是主線方法。** `algs/` 底下其餘都是拿來比較的 baseline。一個叫
`maskgit_routenet` 的早期實作在 2026-08 連同它的結果一起移除了。註解裡提到它的地方是在
說明某段邏輯的來源，不是還活著的程式碼。

## Repo 佈局

```text
main.py test_single_tm.py
                    你會執行的（見「執行」）
run_drl.py          上面兩支 spawn 出來的 agent process，跑在自己的終端機、
                    用一般使用者身分；不會手動執行
test_sim_only.py test_single_tm_udp.py
                    未維護，不在論文主線上——見「執行」
algs/               各演算法；algs/__init__.py 的 REGISTRY 把 --alg 對應到類別
config/             env/ algs/ controller/ —— 三個獨立層級
loader/             env_loader.py（拓樸、流量）、train_loader.py（訓練/評估迴圈）
utils/              Ryu app（simple_monitor.py、manager.py）、iperf3 驅動、量測
dataset/            拓樸、TM、候選路徑集，以及產生它們的腳本（含 prepare_dataset.py）
results/            實驗輸出——見「results 佈局」
paper/              圖表產生器 + LP/ILP 下界；索引在 paper/README.md
docs/               方法論筆記；索引在 docs/README.md
scripts/            收尾清理與資料集建置的 shell 輔助腳本
diagnostics/        佐證方法論主張的獨立實驗
A-Traffic-.../      vendored 的 Enero/RouteNet。只用 SAC_PL_KP（gym 環境）跟
                    Enero_datasets（流量資料），其餘都沒有被 import。
```

## 執行

先設好直譯器。系統的 `python3` 沒有 torch，而 `sudo` 不會繼承 conda activate。

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
```

模式由演算法 config 裡的 `sim_training` 決定，**不是**命令列參數。`main.py` 把未設定視為
`False`，所以幾乎所有東西都跑真實環境。`ls2ic_nx` 是唯一 `sim_training=True` 的演算法，
STRIDE 的每個變體都是 `False`。

**模擬訓練路徑未維護。** 程式碼過時，只有 `ls2ic_nx` 走得到，而且沒有跟著 repo 其他部分
更新。一切都以真實 Mininet 為準。

**評估階段的模擬同樣未維護。** `test_single_tm.py --auto` 原本結尾會呼叫
`test_sim_only.py`，讓每個 session 在 `real/` 旁邊多一份 `sim/`。那個呼叫在 2026-08-19
移除了。sim 那邊已經退化成只涵蓋單一 TM，兩半不再量的是同一件事，而且論文的圖表從來
沒有讀過 sim 那半。`test_sim_only.py` 跟 `test_single_tm_udp.py` 還能單獨執行，但它們的
輸出請當作未經驗證，不要放進任何比較。

```bash
# 訓練 —— variant 跟 seed 是兩個獨立旋鈕，沒設就是 (base, 17)
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py \
    --env 32node_144tm_directed --alg stride --seed 18 train
#   -> results/stride/runs/nodiff_32node_s18_<date>_<time>/train/

# 測試 —— --model 指定 checkpoint，session 就落在那次 run 裡面
sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg stride --auto \
    --model results/stride/runs/nodiff_32node_s18_<date>_<time>/train/model
#   -> .../runs/nodiff_32node_s18_<date>_<time>/test/<date>_<time>/
```

**不帶 `--model` 的話測試讀的是 `results/<alg>/model`**，那是每次訓練都會覆寫的即時目
錄，所以被測的是最後一次訓練剛好留下的 checkpoint。要帶 `--model`。

**光靠 `sudo -E` 不能可靠地傳遞變數。** 要用 `sudo -E "VAR=value" "$PY" ...` 的形式。
`STRIDE_VARIANT` 掉了會安靜退回 `base`，於是 run 看起來一切正常但訓練的是錯的東西。

真實環境的 run 會印出 `Building topology ...` 接著 `Controller spawned, wait 30 s ...`，
並啟動 controller 與 agent，預設是 detached tmux session 裡的兩個 window
（`tmux attach -t stride`）。`--terminal gnome|inline` 可以換 backend，只有
gnome 需要 DISPLAY 與 sudo 的環境變數轉發。

收尾清理用 `./scripts/clean.sh`，不要用 `killall` + `mn -c`。Ryu 是以 `python` 這個
process name 執行的，`killall ryu-manager` 永遠抓不到它；卡在 `net.stop()` 的 `main.py`
或 `run_drl.py` 也會繼續佔住 controller port。下一次執行就會接到一個停滯的舊 controller，
產出一條看起來合理但毫無價值的曲線。`clean.sh` 會先殺卡住的主程式，硬性驗證 6633/6653
兩個 port 已釋放，修復 `results/` 的擁有者，並清掉 `.drl_done`。

## Config 系統

`config/__init__.py` 在 import 時掃描 `config/{env,algs,controller}/*_config.py`。每個檔
案匯出一個叫 `config` 的 dict，檔名去掉 `_config` 就是命令列的 key。`main.py` 把三個
dict 合併之後才交給 `run_drl.py`。

`config/algs/stride_config.py` 的結構跟其他檔不同。`_BASE` 是論文報告的主線設定，
`VARIANTS` 裡每一項只列出該 ablation 改了什麼，通常就一行。`build_config()` 讀
`STRIDE_VARIANT`（預設 `base`）。變體名稱不存在會直接拋錯。seed 不是 config 旋鈕，
它是 main.py / test_single_tm.py 的 `--seed`，所有演算法一律適用，預設 17。

**變體只描述架構，僅此而已。** topology 來自 `--env`，seed 來自 `--seed`，所以不存
在 `..._32node_seed18` 這種變體——那個組合是一次 run，不是一個設定。

另外兩個測試期覆寫跟變體正交，所以同一個 checkpoint 可以用好幾種方式評估：
`STRIDE_EVAL_SAMPLE`（greedy 或 sampled 解碼）與 `STRIDE_ATTN_KERNEL`。

## results 佈局

```text
results/<alg>/
├── runs/<variant>_<topology>_s<seed>_<date>_<time>/
│   ├── train/   model/ config.json output_*.txt step_time.txt ...
│   └── test/<date>_<time>[_sampled]/
│       ├── ckpt.txt          checkpoint 路徑與 sha256、解碼、variant/topo/seed
│       └── real/<tm_id>/     Mininet 量測（旁邊原本有一份 sim/，
│                              2026-08-19 刪除）
└── Metrics/ model/ net_info*.csv drl_paths.json ...
           即時交換區——執行期間 Ryu controller 與 agent 之間以檔案溝通的管道。
           不是歸檔。
```

一次 run 擁有它自己的測試，所以每個結果都自帶出處：checkpoint 就在那個 session 的上一
層。一次訓練可以有多個測試（重跑、或換一種解碼），而 `ckpt.txt` 記下每個測試實際載入的
是哪個 checkpoint——因為一個 session 確實可能去評估別處的 checkpoint。

run 的名字同時就是重現它的指令：variant 對應 `STRIDE_VARIANT`，topology 對應 `--env`，
seed 對應 `--seed`。名字裡沒有主機資訊，那寫在歸檔的 `config.json` 裡，連同起訖
時間。

Controller 把 link 統計寫進交換區，agent 讀取，兩者之間沒有 socket。

論文的圖表腳本會把它讀的 session 直接寫死在原始碼裡，所以每張圖來自哪次 run 是看得見的。

### git 裡有什麼，以及新 run 怎麼處理

`results/*/runs/` 有追蹤。論文背後的每一次 run 都在 repo 裡 —— 訓練 log、測試 session、
checkpoint —— 所以 clone 下來不必先把實驗重跑一遍，就能重建每張圖、重新評估每個
checkpoint。旁邊的即時暫存區（`results/<alg>/Metrics/`、`net_info*.csv` 等）不追蹤，
因為每次 run 都會覆寫它，它屬於哪一次 run 無從得知。

**新的 run 預設不推上去。** 沒有規則忽略它們，所以 `git status` 會把每個新 run 顯示成
一行未追蹤目錄。除非那次 run 值得發布，否則就讓它留在那裡。

值得發布的時候，**整個 run 目錄一起 commit**：

```bash
git add results/stride/runs/<name>          # 加目錄，不要加裡面的個別檔案
```

不要只加 log 不加 checkpoint。沒辦法重新評估的結果不算證據，而兩半各走各的正是一份
歸檔失去意義的方式。代價是實在的 —— 一個 checkpoint 12 到 32 MB，之後換掉的話新舊兩份
都會永遠留在歷史裡 —— 所以每個 run 決定一次，想清楚再決定。

## 評估指標

每個測試 session 對每個 TM 會在 `real/<tm_id>/` 底下寫出四組指標：`real_directed_test/`
與 `real_test/`（Mininet 的 `tc` 計數器，分方向 / 雙向加總），以及 `sim_directed_test/`
與 `sim_test/`（用同一組選定路徑，以 NetworkX 離線估算出的同一批數字）。

**要讀 directed 那兩個。** undirected 的聚合會把一條 link 雙向的容量加總，於是單向的飽和
會被平均掉。`paper/figures/` 底下所有東西讀的都是
`real_directed_test/<tm_id>_eval_metrics.csv`。

這些名字裡的「NX」就是 NetworkX——那是離線估算器，角色是交叉檢查，永遠不是報告數字。
舊筆記會叫你去讀 `eval/*_NX_directed`，那些是 wandb 的 key，產生它們的函式
（`quick_sim_eval_with_action`）已經跟著 wandb 一起刪掉了。要讀 CSV。

參考點，Geant `tm_scale=3`、directed：ILP 約 0.67、LP 約 0.67、OSPF 約 0.87 最大鏈路使用
率。訓練好的策略應該落在這個區間內，到 1.0 代表比最短路徑還差。

比較不同 run 的時候，**平均值差距小於一個 baseline 標準差就不准說是進步**。先算出
baseline 的離散程度再說。

## 環境

- conda 環境 `stride`，Python 3.8.20，torch 2.0.1+cu118。
- Mininet 在**系統的** `dist-packages`，靠 `site-packages/system_dist_packages.pth` 橋接
  進 conda 環境。環境如果重建，這個檔案要重做，否則每次 real run 都會死在
  `from mininet.net import Mininet`。
- Ryu 4.34 需要對 `ryu/topology/switches.py` 做本地 patch，由
  `scripts/patch_ryu.py` 套用（idempotent、`--check` 驗證、`--revert` 還原）。少了
  patch 不會報錯，link 延遲只會讀成 0。

- 真實實驗需要 `sudo`，以及有 `openvswitch`、`veth`、`sch_netem` 的 kernel。
- 用 `SIGKILL` 殺掉 real run 會讓 `results/` 變成 root 所有；正常結束會自動改回來。

## 從局部看不出來的耦合

以下都是「你正在改的那段程式看起來自成一體，其實不是」的陷阱。它們**都不會拋錯**，只會
產出一個看起來合理的錯誤數字。

**config 三層以 `{**env, **alg, **ctrl}` 合併，但不是所有東西都讀合併後的 dict。**
`main.py` 呼叫 `init_paths(env_cfg, alg_cfg)` 的時機在組出 `merged_cfg` **之前**，而
`utils/init_path.py` 是直接從未合併的 env 層讀 `env_config["k_paths_file"]`。所以從
STRIDE variant 設定的 env 層 key，在 `merged_cfg` 裡是贏的，卻仍然被 `init_paths` 忽略。
結果是交換器依照一組候選路徑鋪 flow rule，模型卻依照另一組做決策，全程沒有錯誤訊息。
env 層的 key 要放在 `config/env/`。

**`dataset/32node_traffic/k_paths.json` 每個 pair 剛好 20 條，而且順序是凍結的。**
K=10/15 是它的前綴，所以不需要新檔案。K=25/30 不可能是——它們用的是
`k_paths_k30_ext.json`，那個檔的前 20 條跟原檔逐字相同（每個 pair 都驗證過），後面接
10 條 Yen 擴充，並且由獨立的 `32node_144tm_directed_k30` env 指定。改成用更大的 K 重新
產生 `k_paths.json` 會讓 tie 的順序改變，等於安靜地重新定義了「K=20」是什麼，論文裡每
一次 run 都會失效。**只能擴充，不能重新產生。**

**`VARIANTS` 的每一項必須把它依賴的東西寫死，不能用繼承的。** 每一項只列出自己的 diff，
所以它需要但沒有寫出來的值是從 `_BASE` 來的——之後只要動了 `_BASE` 的預設值，就會在沒
碰到那一行的情況下改掉那個 ablation 的架構。`flatfc_nomask` 之所以明確寫死
`encoder_pool: "flatten"` 就是這個原因：它是「移除 encoder」的 ablation，而它以前是靠
`_BASE` 預設為 `flatten` 才成立的。

**checkpoint 的形狀綁在 `action_dim` 上。** K=30 的 checkpoint 無法用 K=20 的 config 載
入，action head 的形狀不同。跨著載會大聲失敗，但拿 K=30 的 checkpoint 去**評估**一般
env 是安靜的——env 會截斷回正規的 20 條前綴，於是它退化成一次 K=20 的 run。

## 看起來像壞掉但其實沒有

- **reward curve 完全不能說明網路是不是真的。** reward 跟著 action 走，所以就算底下的
  鏈路量測好幾小時沒變，曲線照樣在動。要看的是 run 歸檔裡的 `measurement.txt` ——
  `stale_seconds` 是「訓練結束前多久 controller 最後寫過量測」，超過幾個監控週期就代表
  agent 一直在讀凍結的檔案。實際遇過一個 run，3009 步裡有 2988 步讀的是十小時前的快照，
  checkpoint 有存、曲線看起來正常。MLU 卡在 1.0 只是提示不是證據 —— 學不會紓解飽和的
  策略長得一模一樣。
- **某一輪監控沒收到 port stats，就會停止更新 metrics 檔案。** controller 仍然照印
  `[Statistics Module Ok]` 與 `[Flow Installation Ok]`（兩者都在檢查之前），而
  `net_info*.csv` 停留在上一輪，讀那些檔案的 agent 就對著一個凍結的網路繼續訓練。
  要看的是 `[monitor] cycle N: 0/32 switches returned port stats` 這行警告，以及
  `results/<alg>/net_info_directed.csv` 的 mtime 是不是還在動。這在負載高的機器上才
  會出現：回覆是非同步的，而在 `PORT_STATS_WAIT` 出現之前，唯一讓它們有時間抵達的
  就是「裝 flow 剛好花了幾秒」。
- **`kpath_init`、`kpath_reset`、`get_link_features`** 是我們加進 vendored gym 環境
  （`A-Traffic-.../SAC_PL_KP/gym-graph/.../environment16.py`）的方法，不是上游程式碼。
  它們是 per-OD K 候選路徑的介面，**所有**演算法都會走到——`train_loader` 對每個演算法
  都呼叫 `kpath_init`，`algs/widest_path.py` 呼叫 `get_link_features`——所以不要把它們
  當成 STRIDE 專屬。它們在 2026-08-19 之前叫 `maskgit_*`，取自那個已移除的演算法，那個
  名字兩個面向都是誤導。
- **`test_single_tm.py` 跟 `diagnostics/`** —— 前者是實驗進入點，後者放獨立的方法論實驗。
  兩者都不是 unit test 套件，這個 repo 沒有 pytest。
- **`stride.py` 裡的 ELP / candidate-frame / transferable-head 程式碼**是休眠的。那些旋鈕
  預設關閉，論文沒有用到。

## 訓練 log

**Weights & Biases 在 2026-08-17 移除。** 沒有程式碼 import 它，沒有 config 啟用它，
`requirements.txt` 裡也沒有。所有訓練 log 跟所有論文圖表現在都來自 `results/`。還看得到
的提及是說明歷史資料出處的 docstring，以及 vendored 的 `A-Traffic-.../SAC_PL_KP` 檔案，
那些不是我們的，也從來沒被 import。

**本地 log 一直都是原始紀錄，wandb 是副本。** `results/<alg>/runs/<run>/train/*.txt` 是
`train_loader` 直接寫出來的。有好幾次 run 當下根本沒有連線上傳，是事後拿這些檔案回放進
wandb 的，所以 wandb 的副本最好也只是相等，有時候還更差——其中一筆因為 sampled-history
讀取而掉了三分之二的點，還在論文圖裡放了好幾週沒被發現。

| 檔案 | 內容 |
| --- | --- |
| `output_all.txt` | 每步的完整 per-pair float 向量；per-pair 平均 × 100 就是 reward 曲線 |
| `output_{bwd,delay,loss}.txt` | reward 分量，**對 pair 加總後四捨五入成整數**——要除以 OD pair 數（geant 506、32node 992）；這個捨入讓一致性上限落在約 0.006% |
| `output.txt`、`training_mlu.txt` | reward 總和、MLU |
| `step_time.txt`、`training_time.txt` | 每步的 wall clock |
| `action_k_{entropy,topfrac,mode}.txt` | 跨 pair 的動作多樣性；entropy 趨近 0 且 topfrac 趨近 1 代表所有 pair 都塌陷到同一條候選路徑 |
| `kdiv.txt`、`top1.txt`、`entropy_px0.txt` | 每個 denoise step 的診斷，一行一個 list |
| `epsilon.txt` | 只有 `epsilon_ini > 0` 時才寫，所以 STRIDE 的正式 run 沒有這個檔 |
| `<loss name>.txt` | `agents.update()` 回傳的每個 key 各一個檔 |

reward 那幾列以下的東西都是 2026-08-17 才加的。比那更早的 run 一個都沒有，它們的每步
計時只以 `paper/figures/timing/train_steps/timing_steps.csv` 裡 `source = wandb_cache` 的形式存在，
無法重新產生。

在上述所有檔案裡，對所有演算法而言，行號 `i` 對應的是訓練的第 `i+1` 步。

出處現在是**結構性**的：測試 session 就住在它所評分的那個 checkpoint 所屬的 run 裡面，
而 `ckpt.txt` 記下它實際載入的東西的 sha256。在 `runs/` 佈局（2026-08-19）之前，這層
關聯只存在於目錄名稱，所以碰到更早的資料，要先確認 `ckpt.txt` 存在再談某個數字的出處。
reward 曲線背後對應的訓練 run 寫在
`paper/figures/reward/make_curves_csv.py::RUN_MAP`。

## 在宣稱「可以了」之前

這個專案有好幾種**安靜的失敗模式**。

- 少了 Ryu patch，每條 link 的延遲都會讀成 0
- `STRIDE_VARIANT` 掉了會退回 `base`（`--seed` 是參數，不會用同樣的方式掉；
  `STRIDE_SEED` 已移除，還設著會直接拋錯）
- 測試沒帶 `--model`，測的是訓練最後留在 `results/<alg>/model` 的那個 checkpoint
- 圖表的 glob 沒解析到，產出的是一張空圖
- undirected 指標會藏住單方向的 link 飽和

要看實際輸出，不要只看 exit code。
