import numpy as np
from typing import Sequence, Dict
from .analyze import soft_match


def scan_scale(peaks: Sequence[float], grid: Sequence[float], cfg: Dict):
    s_min, s_max = cfg.get("SCALE_RANGE", [0.8, 1.2])
    p_min, p_max = cfg.get("PHASE_RANGE", [0, np.pi])
    s_steps = cfg.get("SCALE_STEPS", 5)
    p_steps = cfg.get("PHASE_STEPS", 5)

    best_ratio = -1
    best_s = None
    best_phase = None

    scales = np.linspace(s_min, s_max, s_steps)
    phases = np.linspace(p_min, p_max, p_steps)

    for s in scales:
        scaled_grid = np.array(grid) * s
        for ph in phases:
            shifted = scaled_grid + ph
            ratio = soft_match(peaks, shifted, cfg.get("EPS", 0.05))
            if ratio > best_ratio:
                best_ratio = ratio
                best_s = s
                best_phase = ph
    return {"ratio": best_ratio, "s": best_s, "phase": best_phase}
