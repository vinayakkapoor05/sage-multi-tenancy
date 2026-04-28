from __future__ import annotations

import base64
import subprocess


def capture_rtsp_frame_base64(rtsp_url: str, timeout_s: float = 8.0) -> str:
    """
    Capture one frame from RTSP URL and return base64-encoded JPEG bytes.
    Uses ffmpeg so we avoid extra Python camera dependencies.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; install ffmpeg to enable RTSP snapshots") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"timed out while reading RTSP frame: {rtsp_url}") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg RTSP capture failed: {err or 'unknown error'}") from e

    if not proc.stdout:
        raise RuntimeError("ffmpeg produced empty frame output")

    return base64.b64encode(proc.stdout).decode("ascii")
