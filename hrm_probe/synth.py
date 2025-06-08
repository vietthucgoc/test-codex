import numpy as np
from typing import List, Tuple


def synth_signal(lam: List[List[float]], cfg: dict):
    """Synthesize signal from lambda table."""
    sample_rate = cfg.get("SAMPLE_RATE", 1000)
    t_max = cfg.get("T_MAX", 1.0)
    t = np.linspace(0, t_max, int(sample_rate * t_max), endpoint=False)
    signal = np.zeros_like(t)
    rng = np.random.default_rng()

    for row in lam:
        for l in row:
            phase = rng.uniform(0, 2 * np.pi)
            signal += np.sin(2 * np.pi * l * t + phase)
    return signal, t
