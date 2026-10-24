# PASS / FAIL 功能驗證 Demo

在專案根目錄執行以下指令，產生真實影像檔、逐幀時間戳、profile、manifest，並自動分析與比對預期結果：

```sh
.venv/bin/python -m scripts.demo
```

需要已安裝本專案與 FFmpeg / ffprobe，FFmpeg 必須包含 libx264、libx265、mjpeg encoder。輸入檔生成於 `demos/generated/`，報告生成於 `reports/demo/`。可用 `--output`、`--reports` 指定其他目錄。

| 檔案（相對於 generated） | 預期整體結果 | 驗證重點 |
| --- | --- | --- |
| normal.raw | PASS | RAW；七項檢查全部 PASS |
| normal_h264.h264 | PASS | H.264；七項檢查全部 PASS |
| normal_h265.h265 | PASS | H.265；七項檢查全部 PASS |
| normal_mjpg.mjpg | PASS | MJPEG；七項檢查全部 PASS |
| freeze.raw | FAIL | freeze：連續畫面相同超過門檻 |
| dropped_frames.raw | FAIL | fps：capture 時間戳植入 5 幀間距的缺口 |
| green.raw | FAIL | green：植入大面積綠色區塊 |
| blur.raw | FAIL | blur：植入 Gaussian blur |
| color.raw | FAIL | color：影像偏紅，偏離中性 RGB 參考 |
| ghost.raw | FAIL | ghost：植入 A/B 交替返回的時序異常 |
| truncated.raw | ERROR | decode：檔案長度不足一個完整影格 |

每檔 64×48、48 幀、宣告 30 FPS；損毀檔除外。正常檔使用持續移動的灰階紋理。負向案例以 RAW 隔離壓縮損失，有些異常可能同時觸發其他檢查；自動驗證要求表列目標檢查必須 FAIL 且有事件與截圖。掉幀案例保留完整像素幀數，透過合成 capture timestamp 的缺口驗證 FPS/drop 規則。

`synthetic_reference.yaml` 使用 reference 模式與 `calibrated: true`，只表示這組人工資料的已知判斷基準，**不可直接作為真實 V9 場景的校正設定**。這樣才能驗證確定的 PASS／FAIL，而不是 Generic 模式的 SUSPECT。殘影驗證限於目前的 temporal-return 啟發式，色彩驗證限於 RGB 距離。

## 分組執行

```sh
# 四格式正向案例；預期退出碼 0
.venv/bin/videoqa batch --manifest demos/generated/pass.yaml --output reports/demo/pass

# 六種異常；預期退出碼 1
.venv/bin/videoqa batch --manifest demos/generated/fail.yaml --output reports/demo/fail

# 損毀檔；預期退出碼 2
.venv/bin/videoqa batch --manifest demos/generated/error.yaml --output reports/demo/error
```

開啟各組輸出目錄的 `report.html` 可看計量值、判定與證據截圖；同目錄也有 JSON、CSV、JUnit。

`expectations.json` 記錄各案例預期狀態。`scripts.demo` 將實際結果、每組退出碼、事件及證據檔案與預期比對，寫入 `reports/demo/verification.json`。**驗證器退出碼 0 代表 PASS、FAIL、ERROR 都如預期發生**；退出碼 1 表示至少一項結果不吻合；2 表示生成或驗證工具執行錯誤。

## 播放原始影像

使用選單選擇播放檔案：

```sh
python3 scripts/play_demo.py
```

輸入 1～10 選擇 demo，播放器會以 640×480 循環播放。在終端機輸入 **Q** 後按 **Enter** 可停止播放並返回選單；也可先點選播放器視窗再按 **Q** 或關閉視窗。回到選單後輸入 **0** 離開。加上 `--once` 可只播放一遍後自動返回選單；`--dataset` 可指定同格式的 demo 資料夾。腳本只需 Python 標準函式庫與 FFmpeg 安裝提供的 ffplay／ffmpeg；缺少工具或檔案時會顯示補齊方式。

H.264／H.265 裸碼流循環播放需要 `ffmpeg`，腳本會建立含時間戳與索引的暫存 FFV1／MKV 無損預覽，避免 ffplay 的 `error while seeking`；結束播放時自動清除暫存檔。原始 demo 檔可繼續供分析器驗證。

損毀檔 `truncated.raw` 不列入播放選單；掉幀案例的異常在 CSV，須看分析報告才能驗證。

若系統另有 ffplay，可查看正常或異常 RAW：

```sh
ffplay -f rawvideo -pixel_format bgr24 -video_size 64x48 -framerate 30 demos/generated/green.raw
ffplay demos/generated/normal_h264.h264
```

所有 demo 可重新生成，不需將大型二進位檔加入 Git。程式沿用既有 Golden Dataset 生成器，不增加套件依賴。
