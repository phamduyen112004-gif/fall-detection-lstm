"""Realtime profiling utilities (FPS / RAM / GPU)."""

import time
from dataclasses import dataclass, field
from typing import List

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_OK = True
except Exception:
    _GPU_OK = False


@dataclass
class RuntimeProfiler:
    t0: float = field(default_factory=time.time)
    frames: int = 0
    ram_samples: List[float] = field(default_factory=list)
    gpu_samples: List[float] = field(default_factory=list)

    def update(self):
        self.frames += 1
        ram_mb = psutil.Process().memory_info().rss / (1024**2)
        self.ram_samples.append(float(ram_mb))
        if _GPU_OK:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            self.gpu_samples.append(float(util))

    def summary(self):
        elapsed = max(time.time() - self.t0, 1e-6)
        return {
            "fps_avg": float(self.frames / elapsed),
            "ram_mb_avg": float(sum(self.ram_samples) / len(self.ram_samples)) if self.ram_samples else 0.0,
            "gpu_util_avg": float(sum(self.gpu_samples) / len(self.gpu_samples)) if self.gpu_samples else 0.0,
        }
