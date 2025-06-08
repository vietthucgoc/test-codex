import numpy as np
from scipy.signal import find_peaks
from typing import Sequence


def fft_peaks(signal: np.ndarray, sample_rate: float, height: float):
    """Compute FFT and return peak frequencies above height."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) / n
    peaks, _ = find_peaks(spectrum, height=height)
    return freqs[peaks]


def soft_match(peaks: Sequence[float], grid: Sequence[float], eps: float) -> float:
    """Return ratio of peaks matching grid within epsilon."""
    p_arr = np.asarray(peaks)
    g_arr = np.asarray(grid)

    matches = 0
    for p in p_arr:
        if np.any(np.abs(g_arr - p) <= eps):
            matches += 1
    return matches / len(p_arr) if len(p_arr) else 0.0
