import math
from mpmath import mp
import matplotlib.pyplot as plt


# Generate primes up to N using a simple sieve

def primes_up_to(n: int):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(range(start, n + 1, step))
    return [i for i, is_prime in enumerate(sieve) if is_prime]


# Compute R(t, sigma) = sum_{p <= N} cos(t log p) * p^{-sigma}

def R(t: float, sigma: float, primes):
    total = mp.mpf('0')
    for p in primes:
        total += mp.cos(t * mp.log(p)) * (p ** (-sigma))
    return total


# Compute Re(log zeta(s)) for s = sigma + i t

def re_log_zeta(t: float, sigma: float):
    s = mp.mpf(sigma) + 1j * mp.mpf(t)
    return mp.re(mp.log(mp.zeta(s)))


# Find zeros of a function in [start, end] by detecting sign changes

def find_zeros(func, start: float, end: float, step: float = 0.1):
    zeros = []
    t = start
    prev_val = func(t)
    t += step
    while t <= end:
        val = func(t)
        if prev_val == 0:
            zeros.append(t - step)
        elif prev_val * val < 0:
            # sign change -> root exists in (t-step, t)
            try:
                root = mp.findroot(func, (t - step, t))
                if start <= root <= end:
                    zeros.append(root)
            except (ValueError, ZeroDivisionError):
                pass
        prev_val = val
        t += step
    return zeros


def main():
    mp.dps = 50
    sigma = 0.5
    N = 10 ** 6

    print("Generating primes up to", N)
    primes = primes_up_to(N)
    print(f"Total primes: {len(primes)}")

    print("Finding zeros of Re(log zeta) in [10, 30]...")
    zeros_zeta = find_zeros(lambda t: re_log_zeta(t, sigma), 10, 30, step=0.1)
    print("Zeros of Re(log zeta):", zeros_zeta)

    zeros_R = []
    for guess in zeros_zeta:
        try:
            root = mp.findroot(lambda t: R(t, sigma, primes), guess)
            zeros_R.append(root)
        except (ValueError, ZeroDivisionError):
            pass
    print("Zeros of R(t, sigma):", zeros_R)

    standard_zeros = [14.134725, 21.022040, 25.010858]
    for std, z_zeta, z_R in zip(standard_zeros, zeros_zeta, zeros_R):
        print(f"Standard zero {std:.6f} -> Re(log zeta) zero {z_zeta:.6f}, R zero {z_R:.6f}")

    # Compare number of zeros with Riemann-von Mangoldt formula
    T = 30
    N_T = T / (2 * mp.pi) * (mp.log(T / (2 * mp.pi)) - 1)
    print(f"Number of zeros found: {len(zeros_zeta)}")
    print(f"Riemann-von Mangoldt estimate N(30) ~ {N_T}")

    # Plot functions
    ts = [10 + i * 0.05 for i in range(int((30 - 10) / 0.05) + 1)]
    values_R = [R(t, sigma, primes) for t in ts]
    values_zeta = [re_log_zeta(t, sigma) for t in ts]

    plt.figure(figsize=(10, 6))
    plt.plot(ts, values_R, label=r"R(t, $\sigma$)")
    plt.plot(ts, values_zeta, label=r"Re(log zeta($\sigma+it$))")

    for z in standard_zeros:
        plt.axvline(z, color='k', linestyle='--', alpha=0.5)

    plt.xlabel("t")
    plt.legend()
    plt.title("Comparison of R(t, sigma) and Re(log zeta) for sigma=0.5")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png")
    print("Plot saved to plot.png")


if __name__ == "__main__":
    main()
