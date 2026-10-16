"""
Sending 256-bit packets over a Binary Symmetric Channel (BSC),
with and without Hamming coding (Tasks 2 and 3, plus the bonus).

Run:  python simulate.py            (N = 10,000 packets per p, ~1 minute)
      python simulate.py --n 2000   (quicker run)

Outputs: results/recovery_vs_p.png and results/results.csv
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from hamming import HAMMING_7_4, HAMMING_15_11

K = 256                                   # data bits per packet


def bsc(bits: np.ndarray, p: float, rng) -> np.ndarray:
    """Flip each bit independently with probability p."""
    flips = rng.random(bits.shape) < p
    return bits ^ flips.astype(np.uint8)


def simulate_uncoded(p, n_packets, rng):
    data = rng.integers(0, 2, (n_packets, K), dtype=np.uint8)
    received = bsc(data, p, rng)
    return np.all(received == data, axis=1).mean()


def simulate_coded(code, p, n_packets, rng):
    n_blocks = int(np.ceil(K / code.k))               # 64 for (7,4), 24 for (15,11)
    pad = n_blocks * code.k - K                       # 0 for (7,4), 8 zero-pad bits for (15,11)
    data = rng.integers(0, 2, (n_packets, K), dtype=np.uint8)
    padded = np.concatenate([data, np.zeros((n_packets, pad), dtype=np.uint8)], axis=1)
    blocks = padded.reshape(n_packets, n_blocks, code.k)

    tx = code.encode(blocks)                          # (packets, blocks, n)
    rx = bsc(tx, p, rng)
    decoded = code.decode(rx).reshape(n_packets, -1)[:, :K]   # drop the padding
    return np.all(decoded == data, axis=1).mean()


def theory_uncoded(p, k=K):
    return (1 - p) ** k


def theory_coded(code, p, k=K):
    n_blocks = int(np.ceil(k / code.k))
    block_ok = (1 - p) ** code.n + code.n * p * (1 - p) ** (code.n - 1)
    return block_ok ** n_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="packets per value of p")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ps = np.logspace(-4, -1, 15)                       # 15 log-spaced values, 1e-4 .. 1e-1

    rows = []
    for p in ps:
        u = simulate_uncoded(p, args.n, rng)
        c7 = simulate_coded(HAMMING_7_4, p, args.n, rng)
        c15 = simulate_coded(HAMMING_15_11, p, args.n, rng)
        rows.append((p, u, c7, c15))
        print(f"p={p:.2e}  uncoded={100*u:6.2f}%  Hamming(7,4)={100*c7:6.2f}%  Hamming(15,11)={100*c15:6.2f}%")
    rows = np.array(rows)

    os.makedirs("results", exist_ok=True)
    np.savetxt("results/results.csv", rows, delimiter=",",
               header="p,uncoded,hamming74,hamming1511", comments="")

    # ---------- figure ----------
    p_fine = np.logspace(-4, -1, 400)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(p_fine, 100 * theory_uncoded(p_fine), color="tab:red", label="Uncoded – theory $(1-p)^{256}$")
    ax.plot(p_fine, 100 * theory_coded(HAMMING_7_4, p_fine), color="tab:blue",
            label="Hamming(7,4) – theory $[(1-p)^7+7p(1-p)^6]^{64}$")
    ax.plot(p_fine, 100 * theory_coded(HAMMING_15_11, p_fine), color="tab:green", ls="--",
            label="Hamming(15,11) – theory $[(1-p)^{15}+15p(1-p)^{14}]^{24}$")
    ax.plot(rows[:, 0], 100 * rows[:, 1], "o", color="tab:red", label="Uncoded – simulation")
    ax.plot(rows[:, 0], 100 * rows[:, 2], "s", color="tab:blue", label="Hamming(7,4) – simulation")
    ax.plot(rows[:, 0], 100 * rows[:, 3], "^", color="tab:green", label="Hamming(15,11) – simulation")
    ax.set_xscale("log")
    ax.set_xlabel("Bit-flip probability p")
    ax.set_ylabel("Recovered packets [%]")
    ax.set_title(f"256-bit packets over a BSC (N = {args.n:,} packets per point)")
    ax.set_ylim(-3, 103)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/recovery_vs_p.png", dpi=150)
    print("\nSaved results/recovery_vs_p.png and results/results.csv")


if __name__ == "__main__":
    main()
