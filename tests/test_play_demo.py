"""Exercise menu navigation and player invocation without opening GUI windows."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "play_demo.py"


def test_menu_invokes_player_safely_and_recovers_from_invalid_input(tmp_path):
    dataset = tmp_path / "demo files $(unused)"
    dataset.mkdir()
    (dataset / "green.raw").write_bytes(b"demo")
    (dataset / "normal_h264.h264").write_bytes(b"demo")
    log = tmp_path / "calls.jsonl"
    player = tmp_path / "ffplay"
    player.write_text(f"#!{sys.executable}\nimport json, sys\nwith open({str(log)!r}, 'a') as f:\n f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    player.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset), "--once"],
        input="bad\n99\n1\n6\n2\n0\n", text=True, capture_output=True, check=False,
        cwd=tmp_path, env={**os.environ, "PATH": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert "請輸入 0～10" in result.stdout
    assert "檔案不存在" in result.stdout
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert calls[0][-1] == str(dataset / "green.raw")
    assert calls[0][calls[0].index("-video_size") + 1] == "64x48"
    assert calls[0][calls[0].index("-pixel_format") + 1] == "bgr24"
    assert "-autoexit" in calls[0]
    assert calls[1][-1] == str(dataset / "normal_h264.h264")
    assert "-pixel_format" not in calls[1]


def test_missing_ffplay_explains_installation(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT)], input="", text=True,
                            capture_output=True, check=False, env={**os.environ, "PATH": str(tmp_path)})
    assert result.returncode == 2
    assert "brew install ffmpeg" in result.stdout


@pytest.mark.parametrize("stop", ["Q", "interrupt"])
def test_terminal_q_stops_running_player_and_returns_to_menu(tmp_path, stop):
    import pty
    import time

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "normal.raw").write_bytes(b"demo")
    pid_file = tmp_path / "player.pid"
    player = tmp_path / "ffplay"
    player.write_text(f"#!{sys.executable}\nimport os, signal\nfrom pathlib import Path\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\nsignal.pause()\n")
    player.chmod(0o755)
    master, slave = pty.openpty()
    output_path = tmp_path / "menu.log"
    with output_path.open("w+") as output:
        process = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT), "--dataset", str(dataset)], stdin=slave,
            stdout=output, stderr=subprocess.STDOUT, env={**os.environ, "PATH": str(tmp_path)},
        )
        os.close(slave)
        try:
            os.write(master, b"1\n")
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert pid_file.exists(), output_path.read_text()
            if stop == "interrupt":
                process.send_signal(signal.SIGINT)
            else:
                os.write(master, b"Q\n")
            deadline = time.monotonic() + 5
            while output_path.read_text().count("Demo 播放選單") < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert output_path.read_text().count("Demo 播放選單") >= 2, output_path.read_text()
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)
            os.write(master, b"0\n")
            assert process.wait(timeout=5) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), 9)
                except ProcessLookupError:
                    pass
            os.close(master)


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_real_h26x_preview_loops_without_seek_errors(tmp_path, codec):
    import shutil
    import time

    from scripts.generate_golden import encode, normal_frames
    from scripts.play_demo import playback_source

    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe", "ffplay")):
        pytest.skip("FFmpeg / ffprobe / ffplay required")
    source = tmp_path / f"normal.{codec}"
    encode(source, normal_frames(64, 48), codec)
    original = source.read_bytes()
    with playback_source(source, loop=True) as preview:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,nb_read_frames", "-of", "json", str(preview)],
            capture_output=True, text=True, check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["codec_name"] == "ffv1"
        assert int(stream["nb_read_frames"]) == 48
        with subprocess.Popen(
            ["ffplay", "-hide_banner", "-loglevel", "error", "-an", "-loop", "0", str(preview)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        ) as process:
            try:
                # Each fixture is 1.6 seconds: run through two complete loop boundaries.
                time.sleep(3.6)
                assert process.poll() is None, "ffplay exited before looping"
            finally:
                if process.poll() is None:
                    process.terminate()
                _, errors = process.communicate(timeout=5)
        assert "error while seeking" not in errors.decode(errors="replace").lower()
        assert "error" not in errors.decode(errors="replace").lower()
    assert not preview.exists()
    assert source.read_bytes() == original
    with playback_source(source, loop=False) as single_play:
        assert single_play == source
