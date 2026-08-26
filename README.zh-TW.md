# STRIDE — 以擴散模型做 SDN 流量工程的路由決策

> 中文版說明。英文版見 [README.md](README.md)，兩份內容一致。
>
> 只想快點跑起來：[`QUICKSTART.zh-TW.md`](QUICKSTART.zh-TW.md) 是同一套流程，以指令為
> 主軸，每步一兩句話說明。

STRIDE 為每一組來源-目的對（OD pair）從**凍結的 K=20 條候選路徑**中挑一條。所有 pair
由一個離散擴散解碼器**平行且非自迴歸地共同決定**，從全遮罩狀態開始，經過 M 個去噪步逐
步填入並修正。策略以 actor–critic 強化學習在真實 Mininet + Ryu 測試床上訓練。

repo 同時包含論文比較用的所有 baseline（LS2IC、MADQN/PS-DQN、DRSIR、OSPF、widest-path、
adaptive Dijkstra、mean-field，以及靜態 ILP oracle）、評估流程，還有產生論文每一張圖跟
每一張表的腳本。

---

## 1. 環境需求

以下為開發機實測版本。

| 元件 | 版本 | 備註 |
| --- | --- | --- |
| 作業系統 | Ubuntu 20.04 | 系統預設 Python 3.8，Mininet 需要 |
| Python | 3.8.20 | conda 環境，見 §2.2 |
| Mininet | 2.3.1b1 | 系統層安裝，**不在** conda 裡 |
| Open vSwitch | 2.13.8 | 系統 daemon |
| Ryu | 4.34 | pip 安裝，**需要手動 patch**，見 §2.3 |
| PyTorch | 2.0.1+cu118 | CUDA 11.8 |
| GPU | NVIDIA RTX 3060 Ti，8 GB | 論文所有數字都是在這台上量的 |

真實 Mininet 實驗另外需要 `sudo`，以及 kernel 提供 `openvswitch`、`veth`、`sch_netem`
三個模組。一般桌面 Linux 發行版都有，**WSL2 跟精簡版 cloud kernel 沒有**。

真實環境每一步有硬性的時間預算。agent 必須在一個 monitoring period（10 秒）之內完成觀
測、決策與更新，否則就落後於它正在調度的流量。在上面那張卡上，一次路由決策約 22 ms，
32-node 一整步約 4.2 秒。不要假設純 CPU 的機器塞得進這個預算，要先實測再相信它跑出來
的曲線。

---

## 2. 安裝

### 2.1 Mininet + Open vSwitch

兩種裝法，由發行版決定用哪一種，擇一即可。

#### Ubuntu 24.04 以上 —— 用發行版套件

```bash
sudo add-apt-repository universe          # 找不到 mininet 時才需要
sudo apt install mininet openvswitch-switch
sudo systemctl enable --now openvswitch-switch
mn --version                              # 2.3.0
sudo mn --test pingall                    # 驗證
```

**不要**在這裡跑 Mininet 的 `install.sh`：它會先卡 pep8/pycodestyle、再卡 PEP 668，
而每一種繞法都比直接用套件更糟。2.3.0 就夠了 —— 這個 repo 只用到
`Mininet(controller=RemoteController, link=TCLink)` 跟 `addLink(bw=, max_queue_size=)`。

apt 把 Mininet 裝給**系統** Python（24.04 是 3.12），而 §2.2 的 conda 環境是 3.8，
所以 §2.2 的 `.pth` 寫法會指到錯的目錄。改成直接把套件連進來 —— 這樣也只暴露 Mininet
一個，不會把系統為 3.12 裝的所有東西一起攤開：

```bash
conda activate stride
ln -sfn /usr/lib/python3/dist-packages/mininet \
      "$CONDA_PREFIX/lib/python3.8/site-packages/mininet"
python -c "from mininet.net import Mininet; from mininet.link import TCLink; \
           from mininet.node import RemoteController; print('ok')"
```

Mininet 的 python 部分是純 python、目標 3.6+，所以 3.8 的直譯器讀得懂 3.12 的安裝。
編譯的部分 `mnexec` 是 `PATH` 上的執行檔，從來不被 import。§2.2 的 `.pth` 那步跳過，
其餘完全一樣。

#### Ubuntu 20.04 —— 從原始碼

```bash
git clone https://github.com/mininet/mininet src/mininet
cd src/mininet && git checkout -b 2.3.1b1
cd util && ./install.sh -a
sudo mn --test pingall
```

`install.sh -a` 會一併裝 Open vSwitch，需要 20~40 分鐘。裝完再做 §2.2 的 `.pth`。

#### 這些東西最後落在哪

在共用機器上值得知道：

| | 位置 | 你的還是大家的 |
| --- | --- | --- |
| Ryu 與它的 patch | `$CONDA_PREFIX/lib/python3.8/site-packages/ryu/` | **你的** —— 在 env 裡面，env 外一個檔案都不會被動到 |
| Mininet 套件 | `/usr/local/lib/python3.8/dist-packages`（原始碼）或 `/usr/lib/python3/dist-packages`（apt） | 共用 |
| `mnexec`、`ovs-*` | `/usr/bin` | 共用 |
| Mininet 原始碼樹 | 你 clone 的地方 | 你的，而且**裝完就可以刪** |

只有 clone 的位置由你決定 —— 上面的指令放在 repo 底下被 gitignore 的 `src/`，這樣你在
這台機器上加的東西除了系統套件之外都集中在一個目錄，
`install.sh` 跑完那 5 MB 就能刪掉。**安裝本身兩種方式都是系統層級的**：Mininet 要建
network namespace、要接 Open vSwitch，本來就需要 root，這個 repo 改變不了。在共用
server 上，先確認 Mininet 與 OVS 是不是已經裝好了，不要再裝第二份。

### 2.2 Python 環境

```bash
conda create -n stride python=3.8 -y
conda activate stride
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**把 Mininet 橋接進 conda 環境——只有原始碼安裝需要，做一次就好。** 24.04 在 §2.1
的 symlink 已經做過了，直接跳到 §2.3。 Mininet 裝在**系統的**
`dist-packages`，conda 的直譯器看不到它。建立一個 `.pth` 檔把系統路徑接上。之後 Python
每次啟動都會自動讀這個檔，不需要每次執行重做。

```bash
echo /usr/local/lib/python3.8/dist-packages \
  > "$CONDA_PREFIX/lib/python3.8/site-packages/system_dist_packages.pth"
python -c "import mininet; print(mininet.__file__)"   # 必須成功
```

conda 自己的 numpy/torch 在 `sys.path` 上優先級較高，只有 Mininet 會落到系統那份。
**環境如果重建，這個檔案要重做**，否則每次 real 實驗都會死在
`from mininet.net import Mininet`。

### 2.3 Ryu + 延遲量測 patch

Controller 靠 LLDP 封包的往返時間量測每條 link 的延遲，而上游 Ryu 沒有記錄這件事：
`PortData` 只記下封包**送出**的時間，`lldp_packet_in_handler` 從不記回來的時間。

```bash
git clone https://github.com/faucetsdn/ryu src/ryu
pip install "setuptools==58.0.0" wheel "pbr==5.11.1"
pip install ./src/ryu --no-build-isolation
python scripts/patch_ryu.py
```

**不是 `pip install ryu`。** PyPI 上最後一版是 2020 年的 4.34，早於 eventlet 0.30.3
移除 `ALREADY_HANDLED`，裝下去會得到一個「裝得起來但 import 就死」的套件。上游 master
有相容性修正，但之後再也沒發布過，所以 git tree 是唯一能用的來源。

兩個 pin 是各自獨立、而且都必要。`setuptools==58.0.0`：ryu 的 `setup.py` 呼叫
`easy_install.get_script_args`，那個 API 在 setuptools 58 之後被移除；`--no-build-isolation`
則是讓建置看得到這個 pin，而不是用 pip 另外抓的最新版。`pbr==5.11.1`：ryu 寫了
`setup_requires=['pbr']` 且沒指定版本，而 `setup_requires` 是 setuptools 自己的機制、
不是 pip 的 —— 它會在建置當下把最新的 pbr 抓進 `src/ryu/.eggs/`，`--no-build-isolation`
管不到；新版 pbr 會 import `setuptools.extern.tomli`，而 setuptools 要到 61 才開始
vendor 它。事先裝好 pbr 就滿足了需求，不會再去抓。

腳本會在已安裝的套件裡做三處插入 —— `PortData` 加一個 `delay` 欄位、handler 開頭取
接收時間戳、以及填進那個欄位的相減 —— 然後**重新 import 該模組確認改動生效**。它是
idempotent 的，會把未改動的檔案留成 `switches.py.orig`，`--revert` 可以還原。如果檔案
長得不像它預期的 4.34，它會拒絕執行而不是亂猜。

隨時可以驗證：

```bash
python scripts/patch_ryu.py --check      # exit 0 = 已 patch
```

> **這是最值得防的一種失敗。** 少了 patch **不會有任何錯誤**。每條 link 的延遲都讀成
> 0，訓練對著一個常數最佳化，而整個過程看起來一切正常。長時間訓練之前先檢查，不要
> 事後才發現。

因為 Ryu 是 pip 套件，這個 patch 只存在於你的 conda 環境裡 —— env 以外不會寫入任何
東西，環境重建就再跑一次腳本。

### 2.4 Controller 路徑

`config/controller/simple_monitor_config.py` 會在 conda 環境裡啟動 `ryu-manager`。
`conda_env` 與 `conda_sh` 是**整個 repo 裡僅有的兩個機器相關設定值**，設成你在 §2.2
建的環境即可，叫什麼名字都行。其餘路徑都從檔案自身位置推導，不需要修改。

那個環境必須同時裝有 torch 與 patch 過的 Ryu，因為兩個子行程都在裡面啟動。

### 2.5 資料集

```bash
python dataset/prepare_dataset.py --topology 32node --tms 144tm
python dataset/prepare_dataset.py --topology 32node --tms 24tm
python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3
```

---

## 3. 架構與目錄結構

### 跑起來是三個行程

一次 real 實驗會開三個 tmux 視窗。你啟動的是 `main`，另外兩個由它開出來。

```text
                       main.py  (sudo)
             建拓樸、送 iperf3 流量、最後收拾
                     |                  |
                  開 |                  | 開
                     v                  v
              +------------+      +------------+
              | controller |      |    drl     |
              |ryu-manager |      | run_drl.py |
              +------------+      +------------+
                     |                  ^
                     |   net_info_directed.csv
                     |   paths_metrics.json
                     +----------------->+
                     ^                  |
                     +--drl_paths.json--+
```

| 視窗 | 是什麼 | 負責 |
| --- | --- | --- |
| `main` | `main.py`，需要 sudo | 建拓樸、把另外兩個開起來、驅動 iperf3 流量、結束時收拾 Mininet 與 `ryu-manager`，並把 `results/` 的擁有者改回你 |
| `controller` | `ryu-manager --observe-link`，載入 `utils/` 底下那些 app | 量測網路（用量、延遲、丟包）、維護拓樸、把 agent 選的路徑裝成流表 |
| `drl` | `run_drl.py`，就是 agent | 讀鏈路狀態、為每個源-目標對選一條候選路徑、訓練 |

**三個行程之間沒有 socket，全部靠檔案。** controller 把量測寫進
`results/<alg>/net_info_directed.csv` 與 `paths_metrics.json`，agent 把決策寫進
`drl_paths.json`，controller 再照著裝流表。這個交換區就是 `scripts/clean.sh` 清的東西，
也是好幾個坑的來源（見 `docs/controller_stops_measuring.md`）。

先後順序。

1. `main` 建 Mininet 拓樸
2. 開 `controller`，然後**等 30 秒** —— 讓拓樸發現與第一輪 port 統計先完成
3. 開 `drl`
4. `main` 開始送 iperf3 流量，一直循環到 agent 寫下 `.drl_done`
5. `main` 收拾 Mininet 與 `ryu-manager`，把 `results/` 的擁有者改回你

錯誤只會出現在**它自己那個 pane**。controller 死掉的話 `main` 與 `drl` 都不會有任何提示，
所以看起來還在跑不代表還在量。

### 目錄

```text
main.py                  進入點，建拓樸、啟動 controller 與 agent
run_drl.py               由 main.py 在 agent process 內啟動
test_single_tm.py        對單一 traffic matrix 做評估（真實 Mininet）

                         --- 以下不在論文主線上，保留供參考 ---
test_sim_only.py         模擬側評估。已無維護，見 §4。
test_single_tm_udp.py    UDP 探針量測實驗，不是論文採用的 delay/loss 方法。
                         一次性的嘗試，之後沒有再跑過。

algs/                    各演算法。stride.py 是主方法，其餘為 baseline。
                         algs/__init__.py 的 REGISTRY 把 --alg 對應到類別。
config/                  三個獨立層級 env/ algs/ controller/
loader/                  env_loader.py（拓樸與流量）、train_loader.py（訓練與評估迴圈）
utils/                   Ryu controller app（simple_monitor.py、manager.py）、
                         iperf3 驅動、delay/loss 量測
dataset/                 拓樸、traffic matrix、候選路徑集，以及產生它們的腳本，
                         包含 prepare_dataset.py
results/                 實驗輸出，見 §6
paper/                   論文圖表產生器、LP/ILP 下界，見 §7
docs/                    方法論筆記，索引在 docs/README.md
scripts/                 收尾清理與資料集建置的 shell 輔助腳本
diagnostics/             獨立的方法論實驗。論文流程不會跑到它們，
                         它們的作用是佐證某個設計決策。
A-Traffic-.../           vendored 的 Enero/RouteNet 程式碼。只用到 SAC_PL_KP
                         （gym 環境）跟 Enero_datasets（流量資料）。
```

---

## 4. 執行

### 直譯器

以下每個指令都用**絕對路徑**指定 conda 直譯器。這不是可選的，也不是先
`conda activate` 就能省掉的。`sudo` 會用自己的 `secure_path` 蓋掉 `PATH`，所以在 sudo
底下打 `python` 一定是系統的 Python 3.8，那份沒有 torch。

用 shell 變數可以把它縮寫成 `"$PY"`，但 shell 變數會跟著 shell 一起消失，開新的視窗
就要重設。所以直接寫進 `~/.bashrc`，之後每一個開起來的終端機都已經設好，不用再打。

```bash
echo 'PY="$HOME/miniconda3/envs/stride/bin/python"' >> ~/.bashrc
```repo 內的腳本用同樣的慣例，都從 `$HOME` 推導直譯器
路徑，所以換機器、換使用者帳號都不用改。

### 真實 Mininet 是唯一維護中的路徑

訓練模式由演算法 config 裡的 `sim_training` 決定，**不是**命令列參數。`main.py` 把未
設定視為 `False`，所以。

| 演算法 | `sim_training` | 模式 |
| --- | --- | --- |
| `ls2ic_nx` | `True` | fluid-queue 模擬器 |
| `stride` | `False` | 真實 Mininet，所有變體皆是 |
| 其餘全部 | 未設定 → `False` | 真實 Mininet |

> **模擬訓練路徑目前沒有在維護。** 只有 `ls2ic_nx` 還接著它，而那份程式碼沒有跟著
> repo 其他部分更新，是過時的。要用的話得自己修。**STRIDE 根本沒有模擬訓練的變體。**
> 本文件從這裡開始的所有內容都以真實 Mininet 為準。

**評估階段的模擬同樣沒有維護。** `test_sim_only.py` 用 fluid-queue 模型對路由決策評
分。它原本會在每次 `--auto` 測試 session 結束時被自動呼叫，在真實量測旁邊寫下一份
`sim/`。那個呼叫已經移除，因為 sim 那半邊已經退化成只跑單一 TM，跟旁邊的 real 結果不再
可比。論文的任何圖表都沒有讀過它。腳本本身還在，也還能單獨執行，但它的輸出請當作未經
驗證的東西看待。

### 訓練

```bash
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py \
    --env 32node_144tm_directed --alg stride --seed 18 train
```

歸檔會寫進 `results/stride/runs/nodiff_32node_s18_<date>_<time>/train/`。兩個變數都
可以省略，預設是 `base` 與 `17`。

這行指令的每個部分都有其存在理由。

| 部分 | 原因 |
| --- | --- |
| `sudo` | Mininet 要建立 network namespace 跟 veth pair，還要掛上 Open vSwitch，這些都需要 root 權限。 |
| `-E` | sudo 預設的 `env_reset` 會丟掉你的環境變數。只有 `gnome` 這個 terminal backend 需要它 —— `DISPLAY`、`XAUTHORITY`、`DBUS_SESSION_BUS_ADDRESS` 要保留下來，否則 gnome-terminal 找不到它的 session bus。預設的 `tmux` backend 完全不需要這些。 |
| `"$PY"`（絕對路徑） | `-E` **不會**保留 `PATH`。sudo 的 `secure_path` 會用一份固定的系統清單把 `PATH` 蓋掉。所以在 sudo 底下直接打 `python` 會找到系統的 Python 3.8，那份沒有 torch、沒有 numpy、也沒有 ryu。這也是為什麼**先 `conda activate stride` 沒有用**——activate 做的事就只是改 `PATH`，而 sudo 把 `PATH` 丟掉了。 |
| `"STRIDE_VARIANT=..."` | 選擇架構。要寫成明確的 `VAR=value` 指派形式，不要只靠 `-E`。這個變數如果掉了，程式會靜默退回 `base`，不會報任何錯，但訓練的是錯的東西。 |
| `--seed` | 選擇 seed，對所有演算法都適用。它是一般參數，`sudo` 不會把它弄丟。不給就是 17。 |

真實模式會印出 `Building topology ...` 接著 `Controller spawned, wait 30 s ...`，
並各開一個終端機給 controller 與 agent。

那兩個開在哪由 `--terminal` 決定：

| | |
| --- | --- |
| `tmux` | 各開一個 window 在 detached 的 `stride` session 裡 —— `tmux attach -t stride` 就能看。不需要 DISPLAY，所以 SSH 進來也能跑，而且 window 會活得比 process 久，當掉的畫面留得住。 |
| `gnome` | 各開一個 gnome-terminal 視窗。需要圖形環境，在 `sudo` 下還需要上面那些變數。 |
| `inline` | 兩個都跑在目前的終端機，輸出會交錯。前兩者都不可用時的退路。 |
| `auto` | 有 tmux 就用 tmux，否則有 DISPLAY 就用 gnome，再否則 inline。這是預設。 |

### 測試

```bash
sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg stride --auto \
    --model results/stride/runs/nodiff_32node_s18_<date>_<time>/train/model
```

`--model` 指定要評估的 checkpoint，session 就寫在那次訓練底下，也就是
`.../runs/nodiff_32node_s18_<date>_<time>/test/<date>_<time>/`。一個參數同時決定這
兩件事，結果不可能被歸檔到別次訓練底下。

**不給 `--model` 的話會讀 `results/<alg>/model`**，那是每次訓練都會覆寫的即時目錄。
這種情況下被測的是最後一次訓練剛好留下的 checkpoint，而 session 本身無從辨認。要嘛給
`--model`，要嘛去看 session 寫下的 `ckpt.txt`，裡面有解析後的路徑與每個 checkpoint
檔案的 sha256。

OSPF、ILP、widest-path 與 DRSIR 沒有 checkpoint 可讀，因此不帶 `--model` 執行，各自
成為獨立的 run。

### 一行跑完整條流程

```bash
./scripts/run_chain.sh                                  # 32node + stride，seed 17
./scripts/run_chain.sh geant_directed stride            # 換拓樸
./scripts/run_chain.sh 32node_144tm_directed stride 18  # 換 seed
STRIDE_VARIANT=M4 ./scripts/run_chain.sh                # 換架構
```

參數是位置式的 —— env、alg、seed —— 各自都有預設，所以不帶參數就是 32-node、
`stride`、seed 17。`STRIDE_*` 覆寫寫在前面當環境變數，腳本會用明確的 `VAR=value`
形式轉交給 `sudo`，因為 `sudo -E` 對這些變數不可靠。

它會跑 `clean.sh` → 訓練 → `clean.sh` → 測試剛訓練出來的 checkpoint → `clean.sh`。
32-node 3000 步大約八個半小時，加上五個測試 traffic matrix 約半小時。

**它自己找得到 checkpoint。** run 目錄名字含時間戳，事前無從得知，所以腳本記下訓練前
`runs/` 有哪些、訓練後比差集。用「最新的那個」會在別的東西碰過舊目錄時安靜地測錯，
而測錯的結果看起來跟測對的一模一樣。新增不是剛好一個就中止，不猜。

開頭問一次 sudo 密碼，之後背景每分鐘 refresh，八小時的訓練不會在你離開之後停在密碼
提示。在 tmux 裡跑，controller 與 drl 會開成旁邊的視窗。

跑完之後看三個地方就知道結果能不能用：

```bash
cat results/<alg>/runs/<run>/train/measurement.txt   # stale_seconds 應該是個位數
ls  results/<alg>/runs/<run>/test/<session>/real/    # 五個 traffic matrix，各 154 個檔
ls  results/_terminal_logs/                          # controller 與 agent 的完整輸出
```

關鍵是 `measurement.txt`。controller 中途停止量測的 run 一樣會跑完全部步數、存下
checkpoint、畫出持續變動的 reward 曲線 —— 因為 reward 跟著 action 走，不是跟著網路走。
`stale_seconds` 是訓練結束前多久最後一次量測落地，超過一個監控週期就代表那段時間
agent 都在對著凍結的檔案訓練。見
[`docs/controller_stops_measuring.md`](docs/controller_stops_measuring.md)。

### 收尾清理

```bash
./scripts/clean.sh              # 或指定 alg：./scripts/clean.sh <alg>
```

`clean.sh` 就是把 `mn -c` 跟那幾個 kill 用真正有效的順序做一遍，然後檢查結果。

1. **先**殺卡住的 `main.py` / `run_drl.py`，再動 daemon
2. `pkill -f ryu-manager`、iperf3、`mn -c`
3. 驗證 6633/6653 兩個 port 確實釋放，沒釋放就大聲中止
4. 把 `results/` 的擁有者改回來（sudo 執行會留下 root 檔案），並清掉 `.drl_done` 標記檔

自己手動打 `killall` + `mn -c` 會漏掉第 1 步跟第 3 步，而那兩步才是關鍵。Ryu 實際上是
以 `python` 這個 process name 執行的，`killall ryu-manager` 永遠抓不到它，卡在
`net.stop()` 的東西也一樣抓不到。殘留會繼續佔住 controller port，導致**下一次執行**接
到一個停滯的舊 controller，產出一條看起來合理但其實是垃圾的訓練曲線，而且全程不會有
任何錯誤訊息。

---

## 5. 設定系統

一次執行由三個彼此獨立的選擇決定，每個對應一個命令列參數。

```bash
"$PY" main.py --env geant --alg stride --ctrl simple_monitor train
```

| 參數 | 決定什麼 | 讀哪個檔 |
| --- | --- | --- |
| `--env` | 用哪個拓樸與流量 | `config/env/geant_config.py` |
| `--alg` | 用哪個演算法 | `config/algs/stride_config.py` |
| `--ctrl` | 用哪個 Ryu controller app | `config/controller/simple_monitor_config.py` |

**檔名就是參數值。** `config/__init__.py` 在 import 時走過這三個目錄，凡是有匯出一個叫
`config` 的 dict 的檔案，就把「檔名去掉 `_config`」註冊成一個可用的 key。沒有任何名單
需要同步維護。把 `config/env/mytopo_config.py` 丟進目錄，`--env mytopo` 就能用；把檔案
刪掉，那個 key 就消失。`main.py` 接著把三個 dict 合併成一個交給 `run_drl.py`。

所以可用的 key 就等於目錄裡實際有哪些檔案。

```bash
ls config/env config/algs config/controller | sed 's/_config\.py//'
```

撰寫當下是。

- **env**：`geant`、`geant_directed`、`32node_24tm`、`32node_144tm`、
  `32node_144tm_directed`、`32node_144tm_directed_k30`
- **alg**：`stride`、`ls2ic`、`ls2ic_dd`、`ls2ic_nx`、`ps_dqn`、`ps_dqn_a`、
  `ps_dqn_dd`、`drsir`、`drsir_dd`、`ospf`、`widest_path`、`adaptive_dijkstra`、
  `meanfield`、`ilp`
- **ctrl**：`simple_monitor`

seed 不屬於上面任何一層，它是命令列的 `--seed`，對所有演算法一視同仁。以前有四個
`*_seed18` 設定檔，內容只有 `"seed": 18` 一行，因為 baseline 沒有自己的覆寫機制。

### STRIDE 變體

`config/algs/stride_config.py` 是兩層設計。`_BASE` 就是論文報告的主線設定，`VARIANTS`
裡每一項只列出該 ablation 改動的部分。

```python
"base":          {},                       # 主線 STRIDE
"M4":            {"iter_steps": 4},        # denoise step 階梯
"k10":           {"action_dim": 10},       # 候選路徑數
"nodiff":        {"iter_steps": 1, "decision_token": "tau_only"},
"critic":        {"encoder_rl_grad_src": "critic"},
```

變體只描述架構。topology 由 `--env` 決定，seed 由 `--seed` 決定，所以不存在
`..._32node_seed18` 這種變體。那個組合是一次 run，不是一個設定。啟用哪個變體由
`STRIDE_VARIANT` 環境變數決定，預設 `base`，名稱打錯會直接拋錯。

```bash
STRIDE_VARIANT=M4 "$PY" main.py --env 32node_144tm_directed --alg stride --seed 18 train
```

另外有兩個跟變體正交的測試期覆寫，讓同一個 checkpoint 可以在不同推論模式下評估，不必
另外定義變體。`STRIDE_EVAL_SAMPLE`（greedy 或 sampled 解碼）與 `STRIDE_ATTN_KERNEL`。

---

### 讓 update 塞進 10 秒：`update_window_chunk`

一次 update 取 `mini_batch_seq` 個 window，每個 window 是 `time_seq` 步**連續**的
transition。連續是必要的 —— encoder 裡的 GRU 要沿時間往前滾，所以 window **內部**的
時間步只能序列跑。但 window **之間**彼此獨立（每個都從 `init_hidden` 的零開始），
可以疊成 batch 一起前向。

`update_window_chunk`（C）就是一次疊幾個：

```
C=1   8 個 chunk，每個 batch=1     ← 序列前向最多
C=2   4 個 chunk，每個 batch=2     ← 預設，為 8 GB 的卡選的
C=4   2 個 chunk，每個 batch=4
C=8   1 個 chunk，batch=8
```

**改 C 不會改變訓練。** 兩條路徑的 loss 加權（`/n_w` 對 `*B/n_w`）讓總和相同，而
`.backward()` 是累加梯度，中間沒有 `zero_grad()`，所以 `opt.step()` 收到的梯度逐位元
相同，只差浮點加法的結合順序。它是**平行度旋鈕，不是超參數** —— `mini_batch_seq` 跟
`time_seq` 才是。

代價是記憶體：batch 疊 B 倍，反向要保留的 activation 就是 B 倍。`update()` 拆成
per-chunk backward 正是為了這個（一次做完的完整圖有 512 次前向，在 8 GB 上會 OOM）。

不用改檔案：`STRIDE_CHUNK=4 sudo -E ...` 就會覆寫（跟 `STRIDE_VARIANT` 同一套機制）。
這種取決於機器的值不該寫死在版控裡 —— 8 GB 的卡該是 2，32 GB 的該更大。

**所以 C 該跟著卡的記憶體調。** 用 `nvidia-smi` 看跑起來之後的用量，還有大量餘裕就往上
加一階再量。記憶體不夠會直接 OOM 中斷，不會安靜地壞掉。

## 6. results 目錄佈局

```text
results/<alg>/
├── runs/<variant>_<topology>_s<seed>_<YYYYMMDD>_<HHMMSS>/
│   ├── train/          model/ config.json output_*.txt *_loss.txt
│   │                   training_mlu.txt step_time.txt
│   └── test/<YYYYMMDD>_<HHMMSS>[_sampled]/
│       ├── ckpt.txt    checkpoint 路徑與 sha256、解碼模式、variant/topology/seed
│       └── real/<tm_id>/   Mininet 量測。旁邊原本有一份 sim/，
│                           2026-08-19 刪除，見 §4
└── Metrics/ model/ net_info*.csv drl_paths.json ...
                        執行時的即時交換區。這是 Ryu controller 跟 agent 之間
                        以檔案溝通的管道，不是歸檔。
```

一次 run 擁有它自己的測試。產生某個結果的 checkpoint 就在 session 的上一層，這層關係
不需要靠任何命名規則去承載。一次 run 底下可以有多個測試，例如重跑或換解碼模式，而每個
測試都在 `ckpt.txt` 裡記下它實際載入的是哪個 checkpoint，因為一個 session 確實可能去
評估別次 run 的 checkpoint。

run 的目錄名就是重現它的指令。variant 對應 `STRIDE_VARIANT`，topology 對應 `--env`，
seed 對應 `--seed`。session 名稱只在解碼模式非預設時才標註。兩個名稱都不含主機
名，主機資訊連同起訖時間寫在歸檔的 `config.json` 裡。

Controller 把 link 統計寫進交換區，agent 讀取。兩者之間沒有 socket。`runs/` 是在執行或
session 結束時寫下的歸檔，旁邊的其他目錄都是即時交換區。

---

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

## 7. 論文圖表

```text
paper/figures/<topic>/make_*.py     重新產生某一組圖
paper/figures/reward/*.csv          訓練曲線快取，見下方說明
paper/tables/build_paper_table.py   LP/ILP 比較表
paper/figures/algo/                 演算法虛擬碼渲染
```

每支產生器都從自己的檔案位置推導 repo 根目錄，所以整棵樹可以搬移或改名，不必改路徑。

圖表腳本讀的是 `results/<alg>/runs/<run>/test/<session>/real/<tm>/...`，也就是**評估輸出**，不是
訓練 log。reward 系列的圖是例外，它們讀的是訓練 log，透過
`paper/figures/reward/make_curves_csv.py` 產生的兩個 CSV。

那支腳本從 `results/*/runs/*/train/` 底下的 `output_*.txt` 重建 `train_curves.csv` 跟
`components.csv`。那些 txt 才是每次訓練的**原始紀錄**。CSV 掉了或是新增了 run 就重跑
一次，對應關係寫在腳本裡的 `RUN_MAP`。`paper/` 底下沒有任何東西需要網路。

---

## 8. 評估指標

每次測試 session 對每個 traffic matrix 會在 `real/<tm_id>/` 底下寫出四組指標。

| 目錄 | 怎麼來的 | 聚合方式 |
| --- | --- | --- |
| `real_directed_test/` | Mininet，讀交換器上的 `tc` 計數器 | 分方向 |
| `real_test/` | 同樣的量測 | 一條 link 雙向加總 |
| `sim_directed_test/` | NetworkX，用同一組選定路徑離線估算 | 分方向 |
| `sim_test/` | 同樣的估算 | 雙向加總 |

**要讀 directed 那兩個。** undirected 的聚合會把一條 link 雙向的容量加總，於是一條單向
塞爆、反向閒置的 link 平均起來會看起來很正常。論文從頭到尾讀的都是
`real_directed_test/<tm_id>_eval_metrics.csv`，`paper/figures/` 底下每一支腳本裡都寫著
這條路徑。

`sim_*` 是 NetworkX 的估算值。指標名稱裡的「NX」就是 NetworkX 的縮寫，跟網路領域的任何
術語無關。它跟真實量測共用同一組路由決策，差別在於它是把佇列**模型化**而不是量測，所以
它的角色是交叉檢查，不會被當成報告數字。

Geant 在 `tm_scale=3` 且為 directed 時的參考點。ILP 約 0.67，LP 約 0.67，最短路徑
（OSPF）約 0.87 最大鏈路使用率。訓練好的策略應該落在這個區間內，跑到 1.0 代表比單純的
最短路徑還差。

---

## 9. 分析工具

```text
paper/bounds/check_min_mlu.py         每個 TM 的理論最小 MLU（LP 鬆弛或單路徑 ILP）
paper/bounds/edge_lp_bound.py         edge-based 多商品流 LP 下界，不限制候選集
paper/bounds/k_oracle_curve.py        最佳 MLU 隨候選集大小 K 的變化曲線
dataset/extend_k_paths.py        建立 K=30 候選檔，同時保留凍結的 K=20 前綴
```

---

## 10. 已知陷阱

- **`kpath_init`、`kpath_reset`、`get_link_features`** 是我們加進 vendored gym 環境的
  方法，不是上游原本就有的。它們是 per-OD K 候選路徑的介面，**所有**演算法都會走到，
  STRIDE 跟 baseline 都一樣，所以它們跟 STRIDE 沒有專屬關係。
- **用 `SIGKILL` 殺掉 real 實驗**會讓 `results/` 變成 root 所有。用 `chown -R` 修復，
  或直接跑 `clean.sh`。正常結束的話程式會自動改回來。
- **機器太慢**可能讓 controller 錯過 30 秒的啟動視窗，之後 agent process 會出錯。照
  §4 的方式清乾淨再重跑。
- **`sudo -E` 不保證傳遞環境變數**。一律用 `sudo -E "VAR=value" "$PY" ...` 的形式。

---

## 11. 完全移除

順序是從**只屬於你的**東西排到**大家共用的**。前三步在共用機器上都安全，第四步不是。

對照 §2.1 那張表：Ryu 與它的 patch 在 conda env 裡面，砍 env 就跟著走。Mininet 與
Open vSwitch 是系統層級的，這台機器上的其他人也在用同一份。

### 11.1 先確認實驗資料還在別的地方

`results/*/runs/` 底下是每一次 run 的完整歸檔——訓練 log、測試 session、checkpoint。
它有進 git，但**只有推上去的部分才在別的地方**。

```bash
cd ~/stride && git status --short && git log --oneline origin/main..HEAD
```

兩個都沒有輸出才往下走。有輸出就是還有東西只存在這台機器上。

### 11.2 停掉還在跑的東西

```bash
sudo mn -c                                  # 清掉殘留的 namespace 與 bridge
sudo killall -q iperf3 ryu-manager
tmux ls; sudo tmux ls                       # 兩邊都要看，controller 可能在 root 的 session
```

`sudo tmux ls` 要另外看，是因為不在 tmux 裡啟動時，controller 與 agent 會被丟進一個屬於
root 的 detached session，socket 跟你的分開。確認要砍哪些之後再用
`tmux kill-session -t <名稱>`，不要用 `kill-server`，那會連你其他的 session 一起帶走。

### 11.3 repo 與 conda 環境

```bash
rm -rf ~/stride                             # 含 src/ 底下的 Mininet 原始碼樹
conda env remove -n stride
```

Ryu、延遲量測 patch、以及 §2.1 apt 路線建立的那個 Mininet symlink 都在 env 裡面，
一起消失。這台機器上其他的 conda env 不受影響。

### 11.4 Mininet 與 Open vSwitch —— 共用，會影響其他人

> **這一步之前先確認沒有別人在用。** 這兩個是系統套件，不屬於任何單一使用者。
> 只是不想再跑實驗的話，做到 11.3 就夠了。

apt 路線（Ubuntu 24.04 以上）

```bash
sudo systemctl disable --now openvswitch-switch
sudo apt purge mininet openvswitch-switch openvswitch-common \
               openvswitch-pki openvswitch-testcontroller
sudo apt autoremove
```

原始碼路線（Ubuntu 20.04）。`install.sh` **沒有** uninstall，要自己刪。

```bash
sudo rm -rf /usr/local/lib/python3.8/dist-packages/mininet \
            /usr/local/lib/python3.8/dist-packages/mininet-*.dist-info
sudo rm -f  /usr/local/bin/mn /usr/bin/mnexec
```

原始碼路線的 Open vSwitch **也是 apt 裝的**（`install.sh -a` 內部就是呼叫 apt），
所以 OVS 的部分跟上面那段一樣，用 purge。

### 11.5 Open vSwitch 的殘留狀態

purge 不會刪掉資料庫。裡面是 bridge 定義，確定不再用 OVS 才刪。

```bash
sudo rm -rf /etc/openvswitch /var/lib/openvswitch /var/log/openvswitch
```

### 11.6 驗證

```bash
command -v mn mnexec ovs-vsctl     # 三個都不該有輸出
conda env list | grep stride       # 不該有輸出
ls ~/stride                        # No such file or directory
ip -br link | grep -E "^(s[0-9]|ovs)"   # 不該有殘留介面
```

四項都符合就乾淨了。`ip -br link` 還有 `s1`、`s2` 之類的介面，代表 11.2 的
`sudo mn -c` 沒跑或跑失敗，補跑一次。
