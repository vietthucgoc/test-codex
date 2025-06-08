import math
from typing import List, Tuple, Dict


def generate_primes(n: int) -> List[int]:
    """Generate the first n prime numbers."""
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def build_lambda(cfg: Dict) -> Tuple[List[List[float]], List[int]]:
    """Return table of lambda_{p,k} and prime list."""
    n = cfg.get("N_PRIMES", 10)
    k_max = cfg.get("K_MAX", 3)
    primes = generate_primes(n)
    lam = []
    for p in primes:
        row = []
        for k in range(1, k_max + 1):
            # simple placeholder: lambda = 2 * pi * p * k
            row.append(2 * math.pi * p * k)
        lam.append(row)
    return lam, primes
