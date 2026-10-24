#!/usr/bin/env python3
"""Interactive demo player; only Python's standard library and ffplay are needed."""
from __future__ import annotations

import argparse
import select
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

DEMOS = (
    ("normal.raw", "正常 RAW（PASS）"),
    ("normal_h264.h264", "正常 H.264（PASS）"),
    ("normal_h265.h265", "正常 H.265（PASS）"),
    ("normal_mjpg.mjpg", "正常 MJPEG（PASS）"),
    ("freeze.raw", "畫面凍結（FAIL）"),
    ("green.raw", "綠色區塊（FAIL）"),
    ("blur.raw", "模糊（FAIL）"),
    ("color.raw", "偏紅（FAIL）"),
    ("ghost.raw", "交替殘影候選（FAIL）"),
    ("dropped_frames.raw", "時間戳掉幀（FAIL；播放無法呈現時間戳缺口）"),
)
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "demos" / "generated"


@contextmanager
def playback_source(path, loop):
    """Give elementary H.26x a timestamped, seekable lossless loop preview."""
    if not loop or path.suffix not in {".h264", ".h265"}:
        yield path
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("循環播放 H.264/H.265 需要 ffmpeg；MacBook 請執行 brew install ffmpeg")
    with tempfile.TemporaryDirectory(prefix="videoqa_preview_") as directory:
        preview = Path(directory) / "preview.mkv"
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-framerate", "30", "-i", str(path), "-map", "0:v:0", "-an",
             "-c:v", "ffv1", str(preview)], capture_output=True, text=True, check=False, timeout=30,
        )
        if result.returncode:
            raise RuntimeError(f"無法建立循環播放預覽：{result.stderr.strip()}")
        yield preview


def play(command):
    """Own player lifetime and listen for terminal commands without sharing stdin."""
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True) as process:
        try:
            if not sys.stdin.isatty():
                return process.wait()
            while process.poll() is None:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    line = sys.stdin.readline()
                    if not line or line.strip().lower() in {"q", "quit", "exit"}:
                        return 0
                    print("播放中：在終端機輸入 Q 後按 Enter 可返回選單。", flush=True)
            return process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="以選單選擇 ffplay 播放的 demo")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="demo 資料夾")
    parser.add_argument("--once", action="store_true", help="每次選擇只播放一遍；預設循環播放")
    args = parser.parse_args(argv)
    player = shutil.which("ffplay")
    if player is None:
        print("找不到 ffplay。MacBook 請先執行：brew install ffmpeg")
        return 2
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        print(f"找不到 demo 資料夾：{dataset}")
        print("請在專案根目錄先執行：.venv/bin/python -m scripts.demo")
        return 2
    while True:
        print(f"\nDemo 播放選單 — {dataset}")
        for number, (filename, label) in enumerate(DEMOS, start=1):
            missing = " [缺檔]" if not (dataset / filename).is_file() else ""
            print(f"{number:2}. {label} — {filename}{missing}")
        print(" 0. 離開")
        print("終端機輸入 Q＋Enter，或在播放器視窗按 Q／關閉視窗，可返回選單。")
        print("損毀檔 truncated.raw 不列入播放。")
        try:
            choice = input("請輸入編號：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已離開。")
            return 0
        if choice.lower() in {"0", "q", "quit", "exit"}:
            return 0
        if choice not in {str(number) for number in range(1, len(DEMOS) + 1)}:
            print(f"請輸入 0～{len(DEMOS)} 的編號。")
            continue
        filename, label = DEMOS[int(choice) - 1]
        path = dataset / filename
        if not path.is_file():
            print(f"檔案不存在：{path}")
            print("請在專案根目錄執行 .venv/bin/python -m scripts.demo 重新生成。")
            continue
        if filename == "dropped_frames.raw":
            print("此案例的異常在 capture timestamp CSV；影像固定 30 FPS 播放。請查看 FAIL 報告的 fps 指標。")
        print(f"播放：{filename}（{'單次播放' if args.once else '循環播放'}）")
        print("返回選單：終端機輸入 Q＋Enter；或先點選播放器視窗，再按 Q。", flush=True)
        try:
            with playback_source(path, loop=not args.once) as source:
                command = [player, "-hide_banner", "-loglevel", "error", "-an", "-window_title", label]
                if source.suffix == ".raw":
                    command += ["-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "64x48"]
                if source.suffix != ".mkv":
                    command += ["-framerate", "30"]
                command += ["-vf", "scale=640:480:flags=neighbor"]
                command += ["-autoexit"] if args.once else ["-loop", "0"]
                command.append(str(source))
                code = play(command)
            if code:
                print(f"ffplay 播放失敗（退出碼 {code}），請查看上方錯誤。")
        except OSError as exc:
            print(f"無法啟動 ffplay：{exc}")
        except (RuntimeError, subprocess.SubprocessError) as exc:
            print(f"播放準備失敗：{exc}")
        except KeyboardInterrupt:
            print("\n播放已中止，返回選單。")


if __name__ == "__main__":
    raise SystemExit(main())
