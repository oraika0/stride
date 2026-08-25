# 快速開始

> 英文版見 [QUICKSTART.md](QUICKSTART.md)，兩份內容一致。

只有指令，安裝也包含在內。每步用一兩句話說明它做什麼、留下什麼。想知道為什麼要這樣
做，看 [`README.zh-TW.md`](README.zh-TW.md)。

乾淨 clone 可以跑完全部。`results/*/runs/` 底下的實驗歸檔在 repo 裡 —— 訓練 log、測試
session、checkpoint 都在 —— 所以圖表不必先把實驗重跑一遍就能重建。

> **1–7 每台機器做一次，8–11 是每次實驗的循環。** 第 2 步有兩條路線，二選一，
> 看你的 Ubuntu 版本。完整說明、以及在共用機器上東西各自落在哪，見
> [`README.zh-TW.md` §2](README.zh-TW.md)。

## 1. 取得 repo

**建議整段流程都在 tmux 裡跑。** SSH 斷線不會殺掉安裝，捲軸不會被清掉，而且需要把輸出
貼給別人（或給 AI 看）時可以直接 dump 成檔案，不必手動複製終端機：

```bash
tmux new -s setup
```

之後任何時候 `Ctrl-b d` 離開、`tmux attach -t setup` 回來。


```bash
git clone https://github.com/oraika0/stride.git ~/stride
cd ~/stride
```

底下（以及整份文件）的指令**預設 cwd 是 repo 根目錄** —— 只有 Mininet 那段會離開，
所以它自己會走回來。`apt`、`conda`、`ln -s` 三個不在意你在哪，其餘都在意。

## 2. Mininet + Open vSwitch

**兩條路線，照你的 Ubuntu 選一條，不是兩條都做。** 選了哪一條也決定第 4 步要用哪一行，
記著。

### apt 路線 —— Ubuntu 24.04 以上

用發行版套件，不要跑 Mininet 的 `install.sh`。

```bash
sudo apt install mininet openvswitch-switch
sudo systemctl enable --now openvswitch-switch
dpkg -l mininet openvswitch-switch | grep ^ii        # 兩行都在 = 裝好了
```

**公用機器上，最後那行要先跑。** 兩個都已經在的話就跳過安裝 —— `apt install` 會順手
把已裝的 Open vSwitch 升級並重啟服務，別人正在跑的實驗會被打斷。

### 原始碼路線 —— Ubuntu 20.04

`-s` 讓依賴的 clone 一起放進 `src/`，而不是散在 `$HOME`，而且必須寫在 `-n` `-v` 前面。

```bash
git clone https://github.com/mininet/mininet src/mininet
SRC="$PWD/src"
(cd src/mininet/util && ./install.sh -s "$SRC" -nv)
```

那對括號是刻意的 —— 它讓 `cd` 只在子 shell 裡生效，跑完你還在 repo 根目錄。`src/` 已經在
`.gitignore` 裡，不會被 commit 進去。

## 3. Python 環境

兩條路都一樣。

```bash
conda create -n stride python=3.8 -y
conda activate stride
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 4. 讓 env 看得到 Mininet

apt 是裝給系統 Python 3.12，原始碼安裝是進系統的
`dist-packages`，兩種 conda 直譯器都看不到。照你第 2 步選的路線挑一行：

```bash
# 24.04 (apt)：把那一個套件連進來
ln -sfn /usr/lib/python3/dist-packages/mininet "$CONDA_PREFIX/lib/python3.8/site-packages/mininet"

# 20.04 (原始碼)：把系統路徑加進 sys.path
echo /usr/local/lib/python3.8/dist-packages > "$CONDA_PREFIX/lib/python3.8/site-packages/system_dist_packages.pth"
```

## 5. Ryu 與延遲 patch

**要從 git 裝，不能用 `pip install ryu`。** PyPI 上最後一版
（4.34，2020 年）早於 eventlet 0.30.3 的破壞性變更，裝完 `import ryu` 會死在
`ALREADY_HANDLED`。上游 master 有修，但之後再也沒發布過。

```bash
git clone https://github.com/faucetsdn/ryu src/ryu
pip install "setuptools==58.0.0" wheel "pbr==5.11.1"
pip install ./src/ryu --no-build-isolation
python scripts/patch_ryu.py
```

兩個 pin 都不能省，理由不一樣。

**setuptools 58**：ryu 的 `setup.py` 會呼叫 `easy_install.get_script_args`，那個 API 在
setuptools 58 之後被移除。`--no-build-isolation` 是為了讓建置用 env 裡這個舊版，而不是
pip 另外抓的最新版。

**pbr 5.11.1**：光釘 setuptools 不夠。ryu 的 `setup.py` 寫了 `setup_requires=['pbr']`
而且沒指定版本，而 `setup_requires` 是 **setuptools 自己的機制、不是 pip 的** —— 它會在
建置當下把最新的 pbr 抓進 `src/ryu/.eggs/`，`--no-build-isolation` 管不到那裡。新版 pbr 會
`from setuptools.extern.tomli import load`，而 setuptools 要到 61 才開始 vendor `tomli`，
於是建置死在 `ModuleNotFoundError: No module named 'setuptools.extern.tomli'`。事先把 pbr
裝好，需求就已經滿足，不會再去抓。

如果你在加這個 pin 之前已經撞到那個錯，重試前要先砍掉 `src/ryu/.eggs` —— 不然它會直接用
已經抓下來的那份。

延遲 patch 是每條 link 的延遲量得到的前提，**少了它延遲全部讀 0，而且不會有任何錯誤**。

## 6. 產生流量腳本

把 pickle 的 traffic matrix 轉成 Mininet 要重放的 iperf3 腳本，寫到
`dataset/<topo>_traffic/<tm 目錄>/TM-<id>/{Clients,Servers}/`。目錄已存在就跳過。

```bash
python dataset/prepare_dataset.py --topology 32node --tms 144tm
python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3
```

## 7. 檢查

三行都要過才往下走。

```bash
python -c "import torch, mininet; print(torch.__version__, torch.cuda.is_available())"
python scripts/patch_ryu.py --check
sudo mn --test pingall
```

第一行要印出 torch 版本跟 `True`，第二行要印出 `PATCHED`。

## 8. 訓練

**第 8 到 10 步一次跑完**，平常大多數時候都是這樣跑：`clean.sh` → 訓練 → `clean.sh` →
用剛訓練出來的 checkpoint 測試 → `clean.sh`。

```bash
./scripts/run_chain.sh                                  # 32node + stride
./scripts/run_chain.sh geant_directed stride            # 換 env
./scripts/run_chain.sh 32node_144tm_directed stride 18  # 換 seed
STRIDE_CHUNK=4 ./scripts/run_chain.sh                   # 帶覆寫
```

**它自己找得到 checkpoint。** run 目錄名字含時間戳，事前無從得知，所以腳本記下訓練前
`runs/` 有哪些、訓練後比差集 —— 用「最新的那個」會在別的東西碰過舊目錄時安靜地測錯，
而測錯的結果看起來跟測對的一模一樣。新增不是剛好一個就中止，不猜。

開頭問一次 sudo 密碼，之後背景每分鐘 refresh，八小時的訓練不會在你離開之後停在密碼提示。
`STRIDE_*` 那些變數會用明確的 `VAR=value` 形式傳給 sudo，因為 `sudo -E` 對它們不可靠。

這一步剩下的部分跟第 9、10 步，是同三件事分開來跑 —— 給「只想訓練不測試」或「測手上已經有的
checkpoint」用的。下面關於 tmux、輸出與變體的說明兩邊都適用。

### 只跑訓練

**在第 1 步那個 tmux session 裡開一個具名視窗給它**，三個行程就會各佔一個視窗，而你
原本那個 shell 保持空著可以查東西：

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
tmux new-window -n main "sudo -E $PY main.py --env 32node_144tm_directed --alg stride train; exec bash"
```

視窗會叫 `main`，controller 與 drl 隨後加在它旁邊。結尾的 `exec bash` 讓視窗在訓練結束
或當掉之後留著，錯誤訊息才看得到。

`sudo -E` 會把 `$TMUX` 一起帶過去，launcher 看到它就把 controller 與 drl 開成**目前這個
session 的新視窗**，而不是另外建一個 root 的。所以三個都用一般的 `tmux attach` 就看得到，
不需要 `sudo tmux`。

**不在 tmux 裡跑的話**，主行程留在原地，controller 與 drl 會被丟進一個 detached 的
**root** session —— 那個要 `sudo tmux attach -t stride` 才看得到，因為它屬於 root，
socket 跟你的分開（`/tmp/tmux-0/` 對 `/tmp/tmux-1000/`）。

`$PY` 的絕對路徑不可省，也不能換成 `python`：`sudo` 會用自己的 `secure_path` 蓋掉
`PATH`，所以在它底下打 `python` 一定是系統的 Python，那份沒有 torch。

建拓樸、啟動 Ryu controller 與 agent，然後跑 3000 個監控週期（每個 10 秒，約 8 小時）。
正常啟動會印出 `Building topology ...` 接著 `Controller spawned, wait 30 s ...`。

### 在 tmux 裡面怎麼操作

```
Ctrl-b d      離開（訓練繼續跑，不會被中斷）
Ctrl-b n / p  切到下一個 / 上一個視窗
Ctrl-b w      列出所有視窗用選的
Ctrl-b [      進捲動模式往上翻（q 離開）
```

視窗名稱是 `controller` 跟 `drl`。程式結束之後視窗會留著（`remain-on-exit`），所以
**當掉的訊息看得到**，不會一閃就消失 —— 標題列會顯示 `Pane is dead (status N)`。先
`Ctrl-b [` 翻 traceback，看完再交給第 9 步的 `clean.sh` 清掉整個 session。

不想用 tmux 的話，`--terminal gnome` 會各開一個 GUI 視窗，`--terminal inline` 則把兩者
的輸出直接混在目前的終端機。

### 產出

```
results/stride/runs/base_32node_s17_<date>_<time>/train/
├── model/            checkpoint
├── config.json       所有設定，加上主機名與起訖時間
├── output_*.txt      reward 與它的分量，一行一步
└── step_time.txt     每步的 wall clock
```

### 其他寫法

```bash
# 換架構（清單見 README §5）
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py --env 32node_144tm_directed --alg stride train

# 換 seed，用來做 cross-seed error bar
sudo -E "$PY" main.py --env 32node_144tm_directed --alg stride --seed 18 train

# 改跑 baseline
sudo -E "$PY" main.py --env 32node_144tm_directed --alg ls2ic_dd train
```

`STRIDE_VARIANT` 必須是環境變數，而且要寫成明確的 `VAR=value` 指派 —— 光靠 `sudo -E`
會把它弄丟，掉了就**安靜地訓練預設架構**。`--seed` 是一般參數，不會掉。

## 9. 每次跑完收尾

```bash
./scripts/clean.sh
```

每次跑完、每次當掉之後都要跑。手動打 `killall` + `mn -c` 會漏掉卡住的主程式，也不會
驗證 controller port 有沒有釋放 —— 下一次執行就會接到一個停滯的舊 controller，產出
一條看起來合理但毫無價值的曲線，而且全程沒有任何錯誤訊息。

## 10. 測試

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg stride --auto \
    --model results/stride/runs/base_32node_s17_<date>_<time>/train/model
```

用保留的測試 traffic matrix 評估那個 checkpoint，每個 TM 連續 30 個監控週期。`--auto`
會跑完全部；不加就會逐一問你要跑哪個。看進度的方式跟訓練一樣，`sudo tmux attach -t stride`。

給了 `--model`，session 就**寫在擁有該 checkpoint 的那次 run 裡面**：

```
results/stride/runs/base_32node_s17_<date>_<time>/test/<date>_<time>/
├── ckpt.txt                 載入了哪個 checkpoint，附每個檔案的 sha256
└── real/<tm_id>/
    ├── real_directed_test/  ← 論文讀這個
    ├── real_test/           同樣的量測，但雙向加總
    └── sim_*/               NetworkX 離線估算，交叉檢查用
```

要讀 `real_directed_test/<tm_id>_eval_metrics.csv`。undirected 那組會把一條 link 雙向的
容量加總，藏住單方向的飽和。

沒有 checkpoint 可載的 baseline（`ospf`、`ilp`、`widest_path`、`drsir`）本來就不用給
`--model`。它們的 run 目錄只有 `test/` 沒有 `train/`，那正是「沒有訓練階段的 run」該有的
樣子。

## 11. 圖與表

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
for f in paper/figures/*/make_*.py; do (cd "$(dirname "$f")" && "$PY" "$(basename "$f")"); done
"$PY" paper/tables/build_paper_table.py
```

全部從 `results/` 重建，不需要網路也不需要 GPU。檔名就是論文的圖表標題，會原地覆寫。

每支產生器在檔案開頭寫明它讀哪些 session。跑完新的 run 之後，把對應那支指過去、只重跑
它就好：

```bash
cd paper/figures/holdout && "$PY" make_holdout_fig.py
```

reward 曲線是唯一讀訓練 log 而不是測試 session 的一組，它們走
`paper/figures/reward/make_curves_csv.py`，裡面的 `RUN_MAP` 記著每條曲線來自哪次 run。

## 如果 update 塞不進 10 秒

監控週期是 10 秒，模型的推論加更新都要在裡面做完。訓練時看 drl 視窗的
`training_time`，超過就代表這張卡吃不下預設設定。

第一個該調的是 `config/algs/stride_config.py` 的 `update_window_chunk` —— 它決定一次
update 把幾個 window 疊成一批送進 GPU。**改它不會改變訓練結果**（梯度逐位元相同），
只影響平行度與記憶體。預設 `2` 是為 8 GB 的卡選的；卡比較大就往上加一階再量，太大會
直接 OOM 而不是安靜變差。原理見 [`README.zh-TW.md` §5](README.zh-TW.md)。

## 東西放在哪

```
dataset/     拓樸、traffic matrix、凍結的候選路徑 —— 輸入，不會被寫入
results/     所有 run 的產出。runs/<name>/{train,test} 是歸檔，
             results/<alg>/ 底下其餘是 controller 與 agent 交換檔案的即時區
paper/       圖表產生器 —— 見 paper/README.md
docs/        方法論筆記，含走過的死路與失敗原因
```

run 的目錄名就是重現它的指令：`base_32node_s17_...` 等於 `STRIDE_VARIANT=base`、
`--env 32node_144tm_directed`、`--seed 17`。

---

## 完全移除

從只屬於你的排到大家共用的。前兩塊在共用機器上安全，第三塊不是。

**先確認實驗資料還在別的地方。** `results/*/runs/` 是每次 run 的完整歸檔，有進 git，
但只有推上去的部分才在別的地方。

```bash
cd ~/stride && git status --short && git log --oneline origin/main..HEAD
```

兩個都沒有輸出才往下。接著停掉還在跑的東西。

```bash
sudo mn -c
sudo killall -q iperf3 ryu-manager
tmux ls; sudo tmux ls        # 兩邊都看，controller 可能在 root 的 session
```

確認要砍哪些之後用 `tmux kill-session -t <名稱>`，不要 `kill-server`，那會連你其他的
session 一起帶走。然後砍 repo 與 conda 環境。

```bash
rm -rf ~/stride              # 含 src/ 底下的 Mininet 原始碼樹
conda env remove -n stride
```

Ryu、延遲 patch、§4 建立的 symlink 都在 env 裡面，一起消失。其他 conda env 不受影響。

**到這裡為止就夠了**，如果你只是不想再跑實驗。

**Mininet 與 Open vSwitch 是系統套件，這台機器上的其他人也在用同一份。** 要連它們一起
移除，先確認沒有別人在用，指令與各種殘留在 `README.zh-TW.md` §11。

驗證。

```bash
command -v mn mnexec ovs-vsctl     # 三個都不該有輸出
conda env list | grep stride       # 不該有輸出
ip -br link | grep -E "^(s[0-9]|ovs)"   # 不該有殘留介面
```

還有 `s1`、`s2` 之類的介面，代表 `sudo mn -c` 沒跑或跑失敗，補跑一次。

---

## 論文用到的每一次 run

一列一行指令。每一行都是完整的 chain —— clean、訓練、clean、測試、clean —— 所以一列
是一個結果，不是一個步驟。如果 update 在你的卡上塞不進 10 秒，在任何一行前面加
`STRIDE_CHUNK=<n>`（見前一節），它不會改變產出。

下面每一行都是 **seed 17**，也就是預設值。要跑第二個 seed，三個位置參數都要寫出來，
`18` 放最後 —— seed 是第三個參數，不能跳過前兩個單獨給：

```bash
STRIDE_VARIANT=M4 ./scripts/run_chain.sh 32node_144tm_directed stride 18
```

### STRIDE 與被比較的各方法

圖 14、圖 17、表 8、表 9。

其中三個會訓練，所以用 chain：

| 方法 | 32-node | GÉANT |
| --- | --- | --- |
| STRIDE | `./scripts/run_chain.sh` | `./scripts/run_chain.sh geant_directed stride` |
| LS2IC | `./scripts/run_chain.sh 32node_144tm_directed ls2ic_dd` | `./scripts/run_chain.sh geant_directed ls2ic_dd` |
| MADQN | `./scripts/run_chain.sh 32node_144tm_directed ps_dqn_dd` | `./scripts/run_chain.sh geant_directed ps_dqn_dd` |

**另外三個完全沒有訓練階段**，沒有 checkpoint 可以給 test 指，chain 用不上 —— 它會在
檢查 `train/model` 那一步中止。直接跑評估、不帶 `--model`，它自己就會成為一個 run：

| 方法 | 32-node | GÉANT |
| --- | --- | --- |
| DRSIR | `sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg drsir_dd --auto` | `sudo -E "$PY" test_single_tm.py --env geant_directed --alg drsir_dd --auto` |
| OSPF | `sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg ospf --auto` | `sudo -E "$PY" test_single_tm.py --env geant_directed --alg ospf --auto` |
| ILP | `sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg ilp --auto` | `sudo -E "$PY" test_single_tm.py --env geant_directed --alg ilp --auto` |

歸檔的 run 就看得出來：`results/ospf/runs/<run>/` 底下只有 `test`，而 STRIDE 的 run 底下是
`test` 加 `train`。每次跑之前自己先跑一次 `clean.sh`，因為沒有別的東西會幫你跑。

OSPF 與 ILP 是確定性的，只跑一個 seed。DRSIR 是在評估過程中邊跑邊學，所以它沒有獨立的
訓練階段。

LS2IC、MADQN、DRSIR 要用 `_dd` 版本。無向版（`ls2ic`、`ps_dqn`、`drsir`）讀的是把兩個
方向加總過的鏈路指標，會蓋掉單方向的飽和，不是論文拿來比較的對象。

### 去噪步數 M

圖 18、表 10。M=8 是主線設定，所以它是上面的 STRIDE 那一列，不在這裡重複。

| M | 指令 |
| --- | --- |
| 4 | `STRIDE_VARIANT=M4 ./scripts/run_chain.sh` |
| 6 | `STRIDE_VARIANT=M6 ./scripts/run_chain.sh` |
| 10 | `STRIDE_VARIANT=M10 ./scripts/run_chain.sh` |
| 12 | `STRIDE_VARIANT=M12 ./scripts/run_chain.sh` |

### 候選路徑數 K

表 13。K=20 是主線設定。**K=25 與 K=30 要換 `--env`**：
`dataset/32node_traffic/k_paths.json` 每個 pair 剛好 20 條路徑，所以 K=10 與 K=15 是它的
前綴，不需要新東西；K=25 與 K=30 要的比檔案裡有的還多，改讀 `k_paths_k30_ext.json`。
用更大的 K 重新產生 `k_paths.json` 會讓同分的路徑重新排序，等於默默改掉 K=20 的定義，
所以延伸檔另外放，並且用另一個 env 指向它。

| K | 指令 |
| --- | --- |
| 10 | `STRIDE_VARIANT=k10 ./scripts/run_chain.sh` |
| 15 | `STRIDE_VARIANT=k15 ./scripts/run_chain.sh` |
| 25 | `STRIDE_VARIANT=k25 ./scripts/run_chain.sh 32node_144tm_directed_k30 stride` |
| 30 | `STRIDE_VARIANT=k30 ./scripts/run_chain.sh 32node_144tm_directed_k30 stride` |

### 元件消融

圖 19、表 11。每一個只拿掉 STRIDE 的一個部分，其餘不動。

| 拿掉的部分 | 指令 | 做了什麼 |
| --- | --- | --- |
| encoder | `STRIDE_VARIANT=flatfc_nomask ./scripts/run_chain.sh` | 用扁平 `Linear` 取代 attention + PMA，並且 pair mask 全為 1，於是每個 pair 都看到完整的全域鏈路狀態，只剩 per-pair head 能區分它們 |
| diffusion | `STRIDE_VARIANT=nodiff ./scripts/run_chain.sh` | 只解碼一次，且不做去噪步數條件化 |
| actor 梯度 | `STRIDE_VARIANT=critic ./scripts/run_chain.sh` | encoder 只由 critic 塑形 |
