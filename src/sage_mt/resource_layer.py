from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from sage_mt.config import GatewaySettings
from sage_mt.models import Engine, LatencyClass


@dataclass
class _GpuStats:
    utilization_pct: float | None = None
    free_vram_mb: int | None = None
    total_vram_mb: int | None = None
    source: str = "none"


@dataclass
class GPUResourceLayer:
    settings: GatewaySettings
    inflight_by_engine: dict[Engine, int] = field(
        default_factory=lambda: {Engine.vllm: 0, Engine.torch: 0}
    )
    _last_stats: _GpuStats = field(default_factory=_GpuStats)

    def _read_tegrastats_once(self, interval_ms: int = 1000, timeout_s: float = 2.0) -> _GpuStats:
        # Jetson (Orin) often lacks NVML support. tegrastats provides GR3D_FREQ/GPU utilization.
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            line = ""
            try:
                deadline = time.monotonic() + timeout_s
                # tegrastats may emit nothing immediately; keep reading until we get a non-empty line.
                while time.monotonic() < deadline and not line:
                    line = proc.stdout.readline().strip()
                    if not line:
                        time.sleep(0.05)
            except Exception:
                line = ""
            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass

            # Utilization tokens vary a bit by JetPack.
            m_util = re.search(r"(?:GR3D_FREQ|GPU)\s+(\d+(?:\.\d+)?)%", line)
            util = float(m_util.group(1)) if m_util else None

            # VRAM-free/total may appear as FB <used>/<total>MB on some configs.
            m_fb = re.search(r"FB\s+(\d+)\s*/\s*(\d+)(?:MB)?", line)
            if m_fb:
                used = int(m_fb.group(1))
                total = int(m_fb.group(2))
                free = max(0, total - used)
            else:
                free = None
                total = None

            stats = _GpuStats(
                utilization_pct=util,
                free_vram_mb=free,
                total_vram_mb=total,
                source="tegrastats",
            )
            self._last_stats = stats
            return stats
        except Exception:
            return self._last_stats

    def _read_gpu_stats(self) -> _GpuStats:
        # Try NVML first; fallback to nvidia-smi text query.
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats = _GpuStats(
                utilization_pct=float(util.gpu),
                free_vram_mb=int(mem.free / 1024 / 1024),
                total_vram_mb=int(mem.total / 1024 / 1024),
                source="pynvml",
            )
            self._last_stats = stats
            return stats
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=1.0,
            ).strip()
            first = out.splitlines()[0]
            util_s, free_s, total_s = [x.strip() for x in first.split(",")]
            if util_s == "N/A" or free_s == "N/A" or total_s == "N/A":
                raise ValueError("nvidia-smi reported N/A")
            stats = _GpuStats(
                utilization_pct=float(util_s),
                free_vram_mb=int(free_s),
                total_vram_mb=int(total_s),
                source="nvidia-smi",
            )
            self._last_stats = stats
            return stats
        except Exception:
            return self._read_tegrastats_once()

    def can_admit(self, engine: Engine, latency_class: LatencyClass) -> bool:
        stats = self._read_gpu_stats()
        inflight = self.inflight_by_engine[engine]
        if engine == Engine.vllm and inflight >= self.settings.max_inflight_vllm:
            return False
        if engine == Engine.torch and inflight >= self.settings.max_inflight_torch:
            return False

        if stats.utilization_pct is not None and latency_class != LatencyClass.realtime:
            if stats.utilization_pct > self.settings.max_non_realtime_gpu_utilization_pct:
                return False

        if stats.free_vram_mb is not None:
            if latency_class == LatencyClass.realtime:
                return stats.free_vram_mb >= self.settings.reserve_realtime_vram_mb
            return stats.free_vram_mb >= self.settings.min_free_vram_mb

        return True

    def on_start(self, engine: Engine) -> None:
        self.inflight_by_engine[engine] += 1

    def on_finish(self, engine: Engine) -> None:
        self.inflight_by_engine[engine] = max(0, self.inflight_by_engine[engine] - 1)

    def metrics(self) -> dict[str, Any]:
        stats = self._read_gpu_stats()
        return {
            "gpu": {
                "utilization_pct": stats.utilization_pct,
                "free_vram_mb": stats.free_vram_mb,
                "total_vram_mb": stats.total_vram_mb,
                "source": stats.source,
            },
            "engine_inflight": {
                "vllm": self.inflight_by_engine[Engine.vllm],
                "torch": self.inflight_by_engine[Engine.torch],
            },
        }
