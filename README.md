# V9 Offline Video Quality Analyzer — V1

以 Python、FFmpeg / ffprobe、OpenCV、NumPy 實作的離線 CPU 品質分析器。支援 RAW、H.264、H.265、MJPEG（裸碼流或 FFmpeg 可識別容器），提供單檔／批次 CLI、事件截圖，以及 HTML、JSON、CSV、JUnit 報告。

## 安裝與快速使用

需要 Python 3.10+，以及 PATH 中的 `ffmpeg`、`ffprobe`。Ubuntu 可透過 `sudo apt install ffmpeg python3-venv` 安裝系統工具。

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
videoqa analyze --input example.h265 --codec h265 --width 1920 --height 1080 --output reports/single
videoqa batch --manifest dataset_001/manifest.yaml --output reports/run_001
```

`python -m videoqa` 與 `videoqa` 等效。可使用 `--help` 查看完整選項。

退出碼：`0` 所有檢查通過指定 CI 政策、`1` 有 FAIL 或政策不接受的候選／不可判定結果、`2` 輸入／工具／解碼執行錯誤。預設 SUSPECT、UNVERIFIABLE 都阻擋 CI；可加上 `--suspect warn --unverifiable warn`，報告仍保留原始狀態，JUnit 對這些項目標為 skipped。批次會繼續分析其他檔案並保留個別讀取錯誤；manifest／profile 結構錯誤則在執行前停止。

## Manifest 與時間戳契約

所有相對路徑（影片、時間戳、profile）以 manifest 所在資料夾為基準。

```yaml
project: v9_video_qa
expected_fps: 30
profile: profiles/1080p.yaml  # 可省略；使用 generic 預設
files:
  - file_id: raw_1080p
    path: raw/1920x1080.raw
    codec: raw
    width: 1920
    height: 1080
    pixel_format: nv12
    bit_depth: 8
    stride_bytes: 1920
    capture_timestamps: timestamps/raw_1080p.csv
    scenario: moving test chart
  - path: h264/1920x1080.h264
    codec: h264
    width: 1920
    height: 1080
```

`expected_fps`、`profile`、`scenario` 可逐檔覆寫。壓縮影片的寬高可省略並由 ffprobe 探測；若提供則必須符合實際影格。`codec` 必填，且須符合探測結果。未知設定鍵會拒絕，避免拼字錯誤悄悄套用預設。

capture timestamp CSV 使用整數奈秒，逐影格對應，數量必須等於有效解碼影格數：

```csv
frame_index,timestamp_ns
0,0
1,33333333
2,66666667
```

`frame_index` 欄位可省略；若提供，須從 0 連續遞增。時間戳必須嚴格遞增。工具無法驗證 sidecar 是否真的來自擷取硬體，使用者須確保來源可信。**沒有 capture timestamp 就不宣稱 Capture FPS 合格**，僅列出幀數與可取得的 PTS 播放速率，FPS 狀態為 UNVERIFIABLE。裸碼流不以宣告 FPS 補造擷取時間。

## RAW 佈局

V1 支援 8-bit、unpacked 的 NV12、I420、YUYV422、RGB24、BGR24、gray，以及 Bayer8。必須提供 `width`、`height`、`pixel_format`；不會猜測設備的佈局。其他位元深度或 packed 格式會明確拒絕。

- `stride_bytes`：第一 plane 的列長；省略時依像素格式計算緊密列長。
- `plane_strides`：逐 plane 列長，可明確指定有 padding 的色度 plane。
- `plane_offsets`：各 plane 起點，相對於每幀 header 之後的 payload。
- `header_bytes`：**每一幀**的 header 長度（預設 0）。
- `frame_size_bytes`：包含 header、planes 與尾端 padding 的每幀總長度；省略時從佈局計算。
- `packing: unpacked`、`bit_depth: 8` 為預設；截斷幀、重疊 plane、非法 stride 等均回報錯誤。

例如 `--pixel-format nv12 --stride-bytes 2048 --plane-strides 2048 2048 --plane-offsets 0 2211840`。自訂 MJPEG 封裝、未知 RAW layout、packed Bayer10/12 需先取得正式格式規格後再擴充 reader。

## 分析與 profile

預設模式為 Generic；凍結、綠區、殘影候選、低清晰度以 SUSPECT 呈現。FPS/drop 使用 capture timestamp 做客觀判定。Decode 檢查有效影格、解碼日誌與錯誤。各項檢查與事件保存原始計量值、閾值、版本及信心。

```yaml
mode: generic
calibrated: false
roi: null                   # 或 [x, y, width, height]，套用畫質分析
max_events: 20              # 每個 detector 的證據事件上限；仍累計事件總數
fps: {tolerance_ratio: 0.05, gap_factor: 1.5, window_seconds: 1.0}
freeze: {difference_threshold: 0.5, min_seconds: 1.0}
green: {min_ratio: 0.2, hue_min: 35, hue_max: 85, min_saturation: 150, min_value: 40}
ghost: {difference_threshold: 5.0, return_ratio: 0.25}
blur: {min_laplacian_variance: 40.0}
color: {reference_rgb: null, max_rgb_distance: 30.0}
```

用 `--profile profiles/generic.yaml` 載入設定。可只覆寫部分欄位。參考模式須明確設定 `mode: reference` 並在實際場景完成校正後設 `calibrated: true`，品質候選才判 FAIL；範例 profile 不代表設備已校正。

方法與限制：

- Freeze：連續影格平均灰階差與持續秒數；缺時間戳時依 expected_fps 估算秒數並標示來源。靜態場景會產生候選。
- Green：HSV mask 比例及連通區座標；正常綠物體也可能觸發。
- Ghost：三幀時序返回比率啟發式，尋找交替殘影候選；不是完整 optical flow 或保證辨識所有移動殘影。
- Blur：ROI 的 Laplacian 變異數；閾值須依解析度、場景重新校正。
- Color：RGB／HSV／YCrCb 均值；有 reference_rgb 時量測 RGB 歐氏距離。無參考時為 UNVERIFIABLE。V1 **未實作色卡 ΔE00**，不宣稱標準色準合格。

像素標準化為 BGR uint8，沿用 FFmpeg/OpenCV 色彩轉換；目前沒有自訂相機色彩管理或 HDR 色準保證。影格以串流分析；僅保留有限前幀與事件證據，不把整支影片載入記憶體。CPU 處理秒數、吞吐量及 process peak RSS 附於報告；RSS 是整個程序的高水位（包含批次前序檔案），不是單檔增量記憶體。

## 報告與證據

輸出目錄包含 `report.html`、`report.json`、`report.csv`、`junit.xml`，與依輸入序號區隔的證據資料夾。事件包含 frame index、可取得的 timestamp、異常計量值，以及前／當下／後影格；檔案首尾沒有鄰幀時只保存可取得影格。連續候選區段合併，證據數量受 `max_events` 限制。HTML 可直接離線開啟，無 CDN 依賴。

## Golden Dataset 與驗證

```sh
python scripts/generate_golden.py --output golden_dataset/generated
videoqa batch --manifest golden_dataset/generated/manifest.yaml --output reports/golden --suspect warn --unverifiable warn
# 上述資料含刻意植入的 drop，預期退出碼 1。
python scripts/generate_golden.py --output golden_dataset/generated/matrix --matrix
videoqa batch --manifest golden_dataset/generated/matrix/manifest.yaml --output reports/matrix --suspect warn --unverifiable warn
pytest --junitxml=reports/tests.xml
ruff check videoqa tests scripts
python -m compileall -q videoqa
```

生成器提供 normal、freeze、dropped_frames、green、blur、color、ghost。`--matrix` 產生四種格式 × 三種規劃解析度，需 FFmpeg 包含 libx264、libx265、mjpeg encoder。合成資料可重建，避免提交大型影片。`Jenkinsfile` 示範測試與報告封存。

目前沒有實際 V9 的 12 組錄製樣本；合成測試不代表實機格式、擷取時鐘及場景閾值已驗收。V2 CUDA/NVDEC 不在本版實作範圍。

## 明確 PASS / FAIL Demo

```sh
.venv/bin/python -m scripts.demo
```

產生 4 個正常格式檔、6 個異常檔及 1 個損毀檔，並自動驗證每組的退出碼、目標檢查結果及事件證據。詳見 [Demo 操作說明](demos/README.md)。報告位於 `reports/demo/{pass,fail,error}/report.html`，總驗證結果位於 `reports/demo/verification.json`。這組 synthetic reference profile 僅供人工樣本驗證，不代表實際 V9 場景已校正。

### 在 MacBook 上播放 Demo

使用 FFmpeg 附帶的 `ffplay` 播放。若已安裝 Homebrew，可用以下指令安裝 FFmpeg：

```sh
brew install ffmpeg
```

先在終端機切換到本專案根目錄。若尚未產生 demo，先執行上方的 `.venv/bin/python -m scripts.demo`。

建議使用互動式選單選擇要播放的 demo：

```sh
python3 scripts/play_demo.py
```

輸入編號選擇正常或異常案例；播放器預設放大並循環播放，在終端機輸入 **Q** 後按 **Enter**，即可停止播放並返回選單；也可先點選播放器視窗，再按 **Q** 或關閉視窗。在選單輸入 **0** 離開。若只想播放一遍後自動返回：

```sh
python3 scripts/play_demo.py --once
```

腳本使用 Python 標準函式庫、`ffplay` 與 `ffmpeg`，不需啟用虛擬環境。H.264／H.265 裸碼流循環播放時，會先轉成含時間戳與索引的暫存 FFV1／MKV 無損預覽，供 ffplay 正確 seek；播放結束自動清除暫存檔。可使用 `--dataset /path/to/demo` 指定其他 demo 資料夾（須符合目前 demo 的 64×48、BGR24 RAW 與 30 FPS 格式）。

播放正常 H.264 影片：

```sh
ffplay -framerate 30 -autoexit -vf "scale=640:480:flags=neighbor" demos/generated/normal_h264.h264
```

播放綠區異常 RAW；RAW 沒有影片標頭，因此必須指定像素格式、寬高與播放幀率：

```sh
ffplay -f rawvideo -pixel_format bgr24 -video_size 64x48 \
  -framerate 30 -loop 0 -vf "scale=640:480:flags=neighbor" \
  demos/generated/green.raw
```

將 `green.raw` 替換成以下檔名，可查看其他案例：

| 檔名 | 內容 |
| --- | --- |
| `normal.raw` | 正常移動紋理 |
| `freeze.raw` | 畫面凍結 |
| `blur.raw` | 模糊 |
| `color.raw` | 偏紅 |
| `ghost.raw` | 交替殘影候選 |

Demo 原始尺寸為 64×48、長度約 1.6 秒；上述指令放大為 640×480。RAW 範例以 `-loop 0` 循環播放，H.264 範例以 `-autoexit` 單次播放後結束。H.264／H.265 裸碼流直接使用 ffplay 的 `-loop 0` 可能出現 `error while seeking`；需要循環播放時請使用選單腳本。手動執行 ffplay 指令時，請先點選播放器視窗，再按 **Q** 離開。

`dropped_frames.raw` 的異常植入在時間戳 CSV，固定 30 FPS 播放無法呈現掉幀判定；請查看分析報告。`truncated.raw` 是刻意截斷的錯誤輸入，不是正常可播放影片。在 macOS 可直接開啟報告：

```sh
open reports/demo/pass/report.html
open reports/demo/fail/report.html
open reports/demo/error/report.html
```
