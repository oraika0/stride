# STRIDE — 以擴散模型做 SDN 流量工程的路由決策

> 中文版說明。英文版見 [README.md](README.md)，兩份內容一致。
>
> 只想快點跑起來：[`QUICKSTART.zh-TW.md`](QUICKSTART.zh-TW.md) 是同一套流程，以指令為
> 主軸，每步簡短說明。

STRIDE 為每一組來源-目的對（OD pair）從**固定的 K=20 條候選路徑**中挑一條。所有 pair
由一個離散擴散解碼器**平行且非自迴歸地共同決定**，從全遮罩狀態開始，經過 M 個去噪步逐
步填入並修正。策略以 actor–critic 強化學習在真實 Mininet + Ryu 測試平台上訓練。

repo 同時包含論文比較用的 baseline（LS2IC、MADQN/PS-DQN、DRSIR、OSPF，以及靜態 ILP
oracle）、評估流程，還有產生論文每一張圖跟每一張表的腳本。

---

## 1. 環境需求

以下為開發機實測版本。

| 元件 | 版本 |
| --- | --- |
| 作業系統 | Ubuntu 20.04 |
| Python | 3.8.20 |
| Mininet | 2.3.1b1 |
| Open vSwitch | 2.13.8 |
| Ryu | 4.34 |
| PyTorch | 2.0.1+cu118 |
| GPU | NVIDIA RTX 3060 Ti，8 GB |

Python 與 PyTorch 在 conda 環境裡（§2.2），Mininet 與 Open vSwitch 在系統層。Ryu 裝在
conda 環境裡，而且**需要手動 patch**（§2.3）。

真實 Mininet 實驗另外需要 `sudo`，以及 kernel 提供 `openvswitch`、`veth`、`sch_netem`
三個模組。一般桌面 Linux 發行版都有。虛擬化或精簡過的 kernel（WSL、部分 cloud image、
部分容器環境）不一定有，裝之前先確認。

```bash
lsmod | grep -E 'openvswitch|veth|sch_netem'    # 已經載入的
modinfo openvswitch veth sch_netem              # 有沒有這個模組可以載
```

真實環境每一步有硬性的時間預算。agent 必須在一個 monitoring period（10 秒）之內完成觀測、決策與更新，否則就落後於它正在調度的流量。在 RTX 3060 Ti 上，一次路由決策約 22 ms，32-node 一整步約 4.2 秒。換一台機器要實際測一次訓練，並看 `drl` pane 印出的 update time 有沒有落在預算內。

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

`universe` 是 Ubuntu 官方套件庫的四個分區之一（`main`、`restricted`、`universe`、`multiverse`），放的是社群維護的套件，Mininet 在裡面。桌面版預設啟用，部分 server 與 cloud image 不會，那時 `apt install mininet` 會回報找不到套件。這一行把那個分區打開。

`systemctl enable --now` 讓 Open vSwitch 開機自動啟動，並立刻先啟動一次。

Open vSwitch 是兩個常駐行程，`ovsdb-server` 存設定、`ovs-vswitchd` 實際轉封包。Mininet 建 switch 時執行的 `ovs-vsctl` 只是用戶端，透過 `/var/run/openvswitch/db.sock` 對 `ovsdb-server` 下命令，沒有能力把 daemon 叫起來。daemon 不在，`ovs-vsctl` 連不上，Mininet 直接中止並印出 `Error connecting to ovs-db with ovs-vsctl`，所以它必須在 Mininet 執行前就已經在跑。

apt 這一條已經把 Mininet 與 Open vSwitch 都裝好了，**不需要**再跑 Mininet 的 `install.sh`。

裝完接著做 §2.2，那裡會把它接進 conda 環境。

#### Ubuntu 20.04 —— 從原始碼

```bash
git clone https://github.com/mininet/mininet src/mininet
SRC="$PWD/src"
(cd src/mininet/util && ./install.sh -s "$SRC" -nv)
sudo mn --test pingall
```

`-n` 裝 Mininet 的依賴與本體，`-v` 裝 Open vSwitch，兩個就夠。**不需要用 `-a`** ——
那會連 POX、oflops、OpenFlow 參考實作一起裝，這個 repo 一個都用不到。

`-s` 指定依賴的原始碼樹放哪，這裡讓它們一起進 `src/`，而不是散落在 `$HOME`。它**必須寫
在 `-nv` 前面**。使用括弧讓 `cd` 只在子 shell 裡生效，跑完你還在 repo 根目錄。
`src/` 已經在 `.gitignore` 裡。

裝完接著做 §2.2，那裡會把它接進 conda 環境。

### 2.2 Python 環境

```bash
conda create -n stride python=3.8 -y
conda activate stride
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### 直譯器與 Mininet 的對應

Mininet 的套件檔案只有一份，位於系統的 `dist-packages`，不會被複製，也不會在 conda 裡另外裝一份。需要處理的是 conda 直譯器能否找到它。

| 元件 | 直譯器 | Mininet 來源 |
| --- | --- | --- |
| `mn`、`sudo mn -c` 這些命令列工具 | 系統的 `/usr/bin/python3` | 本來就在它的搜尋路徑裡，不用處理 |
| 這個 repo 的 `main.py` | **conda 的** `envs/stride/bin/python` | 要靠一個 symlink 接過去 |
| Ryu controller | conda 的 python | 不需要，它不 import mininet |

`main.py` 會 `from mininet.net import Mininet` 自己把拓樸建起來，而它是用 conda 的直譯器執行的（見 §4），所以缺了這個橋接就會中止在那一行。

**把 Mininet 橋接進 conda 環境，做一次就好。** 兩條路線同一個指令，只有來源路徑不同，照你在 §2.1 選的那條挑一行。

```bash
conda activate stride
S=/usr/lib/python3/dist-packages/mininet            # apt 路線（24.04）
S=/usr/local/lib/python3.8/dist-packages/mininet    # 原始碼路線（20.04）

ln -sfn "$S" "$CONDA_PREFIX/lib/python3.8/site-packages/mininet"
python -c "from mininet.net import Mininet; from mininet.link import TCLink; \
           from mininet.node import RemoteController; print('ok')"
```

`ln` 是建連結的指令。`-s` 建 symbolic link（捷徑，指向另一個路徑），`-f` 是目標已經存在就覆蓋掉，`-n` 是目標本身若為指向目錄的捷徑就直接換掉它、而不是鑽進去建在裡面。所以這一行是「在 conda 的 `site-packages/` 裡放一個叫 `mininet` 的捷徑，指到系統那份」，之後 conda 的 Python `import mininet` 就會走到系統的實體檔案。第二個指令是驗證，印出 `ok` 才算成功。

這個做法不修改 `sys.path`，只是在清單上原本就有的 `site-packages/` 目錄裡新增一個項目。**接進來的只有 `mininet` 這一個套件**，系統目錄裡的其他東西（那裡還有另一份 Ryu 跟 setuptools）不會被 conda 看到。

> **這個捷徑在 conda 環境裡面。** `conda env remove -n stride` 之後重建環境，它會跟著環境一起消失，要再跑一次上面那兩行。系統那邊的 Mininet 不受影響，不用重裝。漏掉的話每次 real 實驗都會中止於 `from mininet.net import Mininet`。

### 2.3 Ryu + 延遲量測 patch

Controller 靠 LLDP 封包的往返時間量測每條 link 的延遲，而上游 Ryu 沒有記錄這件事。
`PortData` 只記下封包**送出**的時間，`lldp_packet_in_handler` 從不記回來的時間。

> **論文採用的 per-link delay 不是 LLDP 這條。** LLDP 在穩態下系統性低估，大約只有真值
> 的三分之一，論文改用 tc backlog 作為 ground truth，見
> [`docs/delay_measurement_issues.md`](docs/delay_measurement_issues.md)。patch 還是要打
> —— controller 這條路徑照樣會算 LLDP delay，沒打就讀成 0。

```bash
git clone https://github.com/faucetsdn/ryu src/ryu
pip install "setuptools==58.0.0" wheel "pbr==5.11.1"
pip install ./src/ryu --no-build-isolation
python scripts/patch_ryu.py
```

**不能使用 `pip install ryu`。** PyPI 上最後一版是 2020 年的 4.34，早於 eventlet 0.30.3
移除 `ALREADY_HANDLED`，裝下去會得到一個能安裝但無法 import 的套件。上游 master
有相容性修正，但之後再也沒發布過，所以 git tree 是唯一能用的來源。

兩個 pin 是各自獨立、而且都必要。`setuptools==58.0.0`：ryu 的 `setup.py` 呼叫
`easy_install.get_script_args`，那個 API 在 setuptools 58 之後被移除。`--no-build-isolation`
則是讓建置看得到這個 pin，而不是用 pip 另外抓的最新版。`pbr==5.11.1`：ryu 寫了
`setup_requires=['pbr']` 且沒指定版本，而 `setup_requires` 是 setuptools 自己的機制、
不是 pip 的 —— 它會在建置當下把最新的 pbr 抓進 `src/ryu/.eggs/`，`--no-build-isolation`
管不到。新版 pbr 會 import `setuptools.extern.tomli`，而 setuptools 要到 61 才開始
vendor 它。事先裝好 pbr 就滿足了需求，不會再去抓。

腳本會在已安裝的套件裡做三處插入 —— `PortData` 加一個 `delay` 欄位、handler 開頭取
接收時間戳、以及填進那個欄位的相減 —— 然後**重新 import 該模組確認改動生效**。它是
idempotent 的，會把未改動的檔案留成 `switches.py.orig`，`--revert` 可以還原。

隨時可以驗證：

```bash
python scripts/patch_ryu.py --check      # exit 0 = 已 patch
```


### 2.4 Controller 路徑

`config/controller/simple_monitor_config.py` 會在 conda 環境裡啟動 `ryu-manager`。
`conda_env` 與 `conda_sh` 是本 repo 僅有的兩個機器相關設定值，設為 §2.2 建立的環境
名稱即可。其餘路徑皆由檔案自身位置推導，不需修改。

那個環境必須同時裝有 torch 與 patch 過的 Ryu，因為兩個子行程都在裡面啟動。

### 2.5 資料集

```bash
python dataset/prepare_dataset.py --topology 32node --tms 144tm
python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3
```

---

### 2.6 安裝後的檔案落點

安裝完成後，三個元件分別落在不同位置。在共用機器上需要留意。

| 元件 | 路徑 | 安裝位置 |
| --- | --- | --- |
| Ryu 與它的 patch | `$CONDA_PREFIX/lib/python3.8/site-packages/ryu/` | conda env，env 以外一個檔案都不會被動到 |
| Mininet 套件 | `/usr/local/lib/python3.8/dist-packages`（原始碼）或 `/usr/lib/python3/dist-packages`（apt） | system |
| `mnexec`、`ovs-*` | `/usr/bin` | system |
| Mininet 原始碼樹 | 你 clone 的地方 | 本機目錄，**裝完就可以刪** |

只有 clone 的位置由你決定。§2.1 的指令放在 repo 底下被 gitignore 的 `src/`，這樣你在這台機器上新增的檔案除了系統套件之外都集中在一個目錄，裝完那 5 MB 就能刪掉。

**Mininet 的安裝本身兩種方式都是系統層級的。** 它要建 network namespace、要接 Open vSwitch，本來就需要 root，這個 repo 改變不了。在共用 server 上先確認 Mininet 與 OVS 有沒有裝過，不要再裝第二份。

---

## 3. 架構與目錄結構

### 跑起來是三個行程

一次 real 實驗會開三個 tmux 視窗。你啟動的是 `main`，另外兩個由它開出來。

```text
                            main.py   (sudo)
                  建拓樸、送 iperf3 流量、最後收拾
                     |                                  |
                  開 |                                  | 開
                     v                                  v
          +-------------------+              +-------------------+
          |    controller     |              |        drl        |
          |    ryu-manager    |              |     run_drl.py    |
          +-------------------+              +-------------------+
                     |                                  ^
                     |    net_info_directed.csv         |
                     |    paths_metrics.json            |
                     +--------------------------------->+
                     |                                  |
                     ^           drl_paths.json         |
                     +----------------------------------+
```

| 視窗 | 行程 | 職責 |
| --- | --- | --- |
| `main` | `main.py`，需要 sudo | 建拓樸、把另外兩個開起來、驅動 iperf3 流量、結束時收拾 Mininet 與 `ryu-manager`，並把 `results/` 的擁有者改回你 |
| `controller` | `ryu-manager --observe-link`，載入 `utils/` 底下那些 app | 量測網路（用量、延遲、丟包）、維護拓樸、把 agent 選的路徑裝成流表 |
| `drl` | `run_drl.py`，就是 agent | 讀鏈路狀態、為每個源-目標對選一條候選路徑、訓練 |

**三個行程之間沒有 socket，全部靠檔案。** controller 把量測寫進
`results/<alg>/net_info_directed.csv` 與 `paths_metrics.json`，agent 把決策寫進
`drl_paths.json`，controller 再照著裝流表。

先後順序。

1. `main` 建 Mininet 拓樸
2. 開 `controller`，然後**等 30 秒** —— 讓拓樸發現與第一輪 port 統計先完成
3. 開 `drl`
4. `main` 開始送 iperf3 流量。agent 到這時才有東西可量，訓練從這裡實質展開，並一直循環
   到 agent 寫下 `.drl_done`
5. `main` 收拾 Mininet 與 `ryu-manager`，把 `results/` 的擁有者改回你

**三個 pane 的錯誤各自獨立，三個都要看。** 任何一個都不會回報另外兩個的狀況。controller
中止時，`main` 與 `drl` 不會輸出任何訊息，並繼續執行。agent 中止時，`main` 會持續送流量，
等待一個不會出現的 `.drl_done`。某個 pane 沒有輸出，不代表該行程仍在運作。

### 目錄

```text
main.py                  進入點，建拓樸、啟動 controller 與 agent
run_drl.py               由 main.py 在 agent process 內啟動
test_single_tm.py        對單一 traffic matrix 做評估（真實 Mininet）

                         --- 以下不在論文主線上，保留供參考 ---
test_sim_only.py         模擬側評估。已無維護，見 §4。
test_single_tm_udp.py    UDP 探針量測，非論文採用的 delay/loss 方法。量測結果用於
                         docs/delay_measurement_issues.md 的 LLDP 偏差比較。

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
```

repo 內的腳本用同樣的慣例，都從 `$HOME` 推導直譯器路徑，所以換機器、換使用者帳號
都不用改。

### 真實 Mininet 是唯一維護中的路徑

模擬路徑在訓練與評估兩端都已停止維護，本文件其餘部分一律以真實 Mininet 為準。訓練模式
由演算法 config 裡的 `sim_training` 決定，未設定即為 `False`，STRIDE 的所有變體都是
`False`。評估側的 `test_sim_only.py` 仍可單獨執行，但其輸出未經驗證，論文的任何圖表都
沒有讀過它。

### 訓練

```bash
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py \
    --env 32node_144tm_directed --alg stride --seed 18 train
```

歸檔會寫進 `results/stride/runs/nodiff_32node_s18_<date>_<time>/train/`。兩個變數都
可以省略，預設是 `base` 與 `17`。

指令各部分的作用如下。

| 部分 | 原因 |
| --- | --- |
| `sudo` | Mininet 要建立 network namespace 跟 veth pair，還要掛上 Open vSwitch，這些都需要 root 權限。 |
| `-E` | sudo 預設的 `env_reset` 會丟掉你的環境變數。只有 `gnome` 這個 terminal backend 需要它 —— `DISPLAY`、`XAUTHORITY`、`DBUS_SESSION_BUS_ADDRESS` 要保留下來，否則 gnome-terminal 找不到它的 session bus。預設的 `tmux` backend 完全不需要這些。 |
| `"$PY"`（絕對路徑） | `-E` **不會**保留 `PATH`。sudo 的 `secure_path` 會用一份固定的系統清單把 `PATH` 蓋掉。所以在 sudo 底下直接打 `python` 會找到系統的 Python 3.8，那份沒有 torch、沒有 numpy、也沒有 ryu。這也是為什麼**先 `conda activate stride` 沒有用**——activate 做的事就只是改 `PATH`，而 sudo 把 `PATH` 丟掉了。 |
| `"STRIDE_VARIANT=..."` | 選擇架構。要寫成明確的 `VAR=value` 指派形式，不要只靠 `-E`。這個變數若未傳入，程式會退回 `base` 且不會報錯，訓練的將是另一個架構。 |
| `--seed` | 選擇 seed，對所有演算法都適用。它是一般參數，`sudo` 不會把它弄丟。不給就是 17。 |

真實模式會印出 `Building topology ...` 接著 `Controller spawned, wait 30 s ...`，
並各開一個終端機給 controller 與 agent。

那兩個開在哪由 `--terminal` 決定：

| 值 | 行為 |
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
`runs/` 有哪些、訓練後比差集。用「最新的那個」會在其他程序碰過舊目錄時測到錯的 checkpoint，
而測錯的結果看起來跟測對的一模一樣。新增不是剛好一個就中止，不猜。

開頭問一次 sudo 密碼，之後背景每分鐘 refresh，八小時的訓練不會在你離開之後停在密碼
提示。在 tmux 裡跑，controller 與 drl 會開成額外的視窗。

controller 中途停止量測的 run 一樣會跑完全部步數、存下 checkpoint、畫出持續變動的
reward 曲線，因為 reward 跟著 action 走，不是跟著網路走。訓練迴圈每一步都會比對讀進來
的量測檔，連續五次內容完全相同就中止，不會讓整條 chain 白跑，見
[`docs/controller_stops_measuring.md`](docs/controller_stops_measuring.md)。

### 收尾清理

```bash
./scripts/clean.sh
```

`clean.sh` 就是把 `mn -c` 跟那幾個 kill 用真正有效的順序做一遍，然後檢查結果。

1. **先**殺卡住的 `main.py` / `run_drl.py`，再動 daemon
2. `pkill -f ryu-manager`、iperf3、`mn -c`
3. 驗證 6633/6653 兩個 port 確實釋放，未釋放則終止
4. 把 `results/` 的擁有者改回來（sudo 執行會留下 root 檔案），並清掉 `.drl_done` 標記檔


---

## 5. 設定系統

一次執行由三個彼此獨立的選擇決定，每個對應一個命令列參數。

```bash
"$PY" main.py --env geant_directed --alg stride --ctrl simple_monitor train
```

| 參數 | 決定什麼 | 讀哪個檔 |
| --- | --- | --- |
| `--env` | 用哪個拓樸與流量 | `config/env/geant_directed_config.py` |
| `--alg` | 用哪個演算法 | `config/algs/stride_config.py` |
| `--ctrl` | 用哪個 Ryu controller app | `config/controller/simple_monitor_config.py` |

**檔名就是參數值。** `config/__init__.py` 在 import 時走過這三個目錄，凡是有匯出一個叫
`config` 的 dict 的檔案，就把「檔名去掉 `_config`」註冊成一個可用的 key。沒有任何名單
需要同步維護。把 `config/env/mytopo_config.py` 放進目錄，`--env mytopo` 就能用。把檔案
刪掉，那個 key 就消失。`main.py` 接著把三個 dict 合併成一個交給 `run_drl.py`。

所以可用的 key 就等於目錄裡實際有哪些檔案。

```bash
ls config/env config/algs config/controller | sed 's/_config\.py//'
```

撰寫當下是。

- **env**：`geant`、`geant_directed`、`32node_24tm`、`32node_144tm`、
  `32node_144tm_directed`、`32node_144tm_directed_k30`
- **alg**：`stride`、`ls2ic`、`ls2ic_dd`、`ps_dqn`、`ps_dqn_a`、`ps_dqn_dd`、
  `drsir`、`drsir_dd`、`ospf`、`widest_path`、`ilp`
- **ctrl**：`simple_monitor`

seed 不屬於上面任何一層，它是命令列的 `--seed`，對所有演算法一視同仁。

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

變體只描述架構。topology 由 `--env` 決定，seed 由 `--seed` 決定。啟用哪個變體由
`STRIDE_VARIANT` 環境變數決定，預設 `base`，名稱打錯會直接拋錯。

```bash
STRIDE_VARIANT=M4 "$PY" main.py --env 32node_144tm_directed --alg stride --seed 18 train
```


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
加一階再量。記憶體不夠會直接 OOM 中斷，不會無聲地產生錯誤結果。

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

## 7. 論文圖表

每個圖表對應哪一支生成器，列在 [`paper/README.md`](paper/README.md)。所有生成器都從
`__file__` 推導路徑，在哪個目錄執行都可以，也都不需要網路。

一次重新產生全部：

```bash
# reward 系列的圖依賴這份快取，要先跑
"$PY" paper/figures/reward/make_curves_csv.py

# 圖
"$PY" paper/figures/reward/make_reward_fig.py              # 圖 13、15
"$PY" paper/figures/reward/make_reward_components_fig.py   # 圖 16
"$PY" paper/figures/holdout/make_holdout_fig.py            # 圖 14、17 與表 8、9
"$PY" paper/figures/denoise_step/make_denoise_step_fig.py  # 圖 18 與表 10
"$PY" paper/figures/ablation/make_ablation_fig.py          # 圖 19 與表 11
"$PY" paper/figures/dataset/make_dataset_figs.py           # 圖 6-11
"$PY" paper/figures/dataset/make_demand_concentration.py   # 圖 12

# 表與理論界
"$PY" paper/figures/k_ablation/make_k_fig.py               # 表 13
"$PY" paper/bounds/k_oracle_curve.py                       # 表 12
"$PY" paper/tables/build_paper_table.py                    # LP/ILP 比較表

# 內文引用的時間數字
"$PY" paper/figures/timing/train_steps/make_timing_table.py
```

三支有額外條件，不在上面那批裡：

| 生成器 | 需要 |
| --- | --- |
| `figures/timing/inference/make_inference_bench.py` | GPU 與 `results/` 底下的 checkpoint |
| `figures/algo/render_algo_stride.py` | LaTeX（`pdflatex`）與 `pdftoppm` |
| `figures/control_delay/crop_control_delay.py` | `pdftoppm`，且裁切的是既有的投影片匯出 |

`figures/timing/train_steps/make_timing_csv.py` 只在新增 run 時才需要跑，它把新的訓練
log 併進已經版控的 `timing_steps.csv`。


圖表腳本讀的是 `results/<alg>/runs/<run>/test/<session>/real/<tm>/...`，也就是**評估輸出**，不是
訓練 log。reward 系列的圖是例外，它們讀的是訓練 log，透過
`paper/figures/reward/make_curves_csv.py` 產生的兩個 CSV。

那支腳本從 `results/*/runs/*/train/` 底下的 `output_*.txt` 重建 `train_curves.csv` 跟
`components.csv`。那些 txt 才是每次訓練的**原始紀錄**。CSV 掉了或是新增了 run 就重跑
一次，對應關係寫在腳本裡的 `RUN_MAP`。`paper/` 底下的程式都不需要網路。

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
飽和、反向閒置的 link 平均後會呈現正常的數值。論文從頭到尾讀的都是
`real_directed_test/<tm_id>_eval_metrics.csv`，`paper/figures/` 底下每一支腳本裡都寫著
這條路徑。

`sim_*` 是 NetworkX 的估算值。指標名稱裡的「NX」就是 NetworkX 的縮寫。它跟真實量測共用同一組路由決策，差別在於它是把佇列**模型化**而不是量測，所以
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
dataset/extend_k_paths.py        建立 K=30 候選檔，同時保留固定的 K=20 前綴
```

---

## 10. 已知陷阱

- **機器太慢**可能讓 controller 錯過 30 秒的啟動視窗，之後 agent process 會出錯。照
  §4 的方式清乾淨再重跑。
- **`sudo -E` 不保證傳遞環境變數**。一律用 `sudo -E "VAR=value" "$PY" ...` 的形式。
- **controller 可能在訓練途中停止更新量測**，而 run 依然會跑完全部步數、存下
  checkpoint、畫出持續變動的 reward 曲線。訓練迴圈會比對每一步讀到的量測檔，連續五次
  內容完全相同就中止（`stall_abort_steps`，設 0 停用），見
  [`docs/controller_stops_measuring.md`](docs/controller_stops_measuring.md)。

---

## 11. 完全卸載

順序是從 **conda 環境**排到**系統套件**。前三步在共用機器上都安全，第四步不是。

對照 §2.1 那張表：Ryu 與它的 patch 在 conda env 裡面，移除 env 就一併移除。Mininet 與
Open vSwitch 是系統層級的，這台機器上的其他人也在用同一份。

### 11.1 確認實驗資料已備份

`results/*/runs/` 底下是每一次 run 的完整歸檔——訓練 log、測試 session、checkpoint。
它有進 git，但**只有推上去的部分才在別的地方**。

```bash
cd ~/stride && git status --short && git log --oneline origin/main..HEAD
```

兩個都沒有輸出才往下走。有輸出就是還有東西只存在這台機器上。

### 11.2 停止執行中的行程

```bash
sudo mn -c                                  # 清掉殘留的 namespace 與 bridge
sudo killall -q iperf3 ryu-manager
tmux ls; sudo tmux ls                       # 兩邊都要看，controller 可能在 root 的 session
```

`sudo tmux ls` 要另外看，是因為不在 tmux 裡啟動時，controller 與 agent 會被放進一個屬於
root 的 detached session，socket 跟你的分開。確認要移除哪些之後再用
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

原始碼路線的 Open vSwitch **也是 apt 裝的**（`install.sh` 的 `-v` 內部就是呼叫 apt），
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
