import yaml
import json
import time
import logging
from pathlib import Path
import pandas as pd

from .primes import build_lambda
from .synth import synth_signal
from .analyze import fft_peaks, soft_match
from .tuner import scan_scale


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hrm")


def build_goldbach_grid(lam, primes, cfg):
    grid = []
    for i, p1 in enumerate(primes):
        for j, p2 in enumerate(primes[i:], start=i):
            for k1 in range(cfg.get("K_MAX", 3)):
                for k2 in range(cfg.get("K_MAX", 3)):
                    f = lam[i][k1] + lam[j][k2]
                    grid.append(f)
    return grid


def main(config_path: str = None):
    cfg_path = config_path or Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    lam, primes = build_lambda(cfg)
    logger.info("lambda table built")

    signal, t = synth_signal(lam, cfg)
    logger.info("signal synthesised")

    sample_rate = cfg.get("SAMPLE_RATE", 1000)
    peaks = fft_peaks(signal, sample_rate, cfg.get("PEAK_HEIGHT", 0.1))
    logger.info("fft peaks computed: %d", len(peaks))

    gold = build_goldbach_grid(lam, primes, cfg)
    ratio_soft = soft_match(peaks, gold, cfg.get("EPS", 0.05))
    logger.info("soft match ratio %.3f", ratio_soft)

    best = scan_scale(peaks, gold, cfg)
    logger.info("best ratio %.3f", best["ratio"])

    Path("logs").mkdir(exist_ok=True)
    pd.DataFrame([best]).to_csv("logs/latest_summary.csv", index=False)
    with open("logs/latest_cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
