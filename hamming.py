"""
Hamming codes (7,4) and (15,11) with a syndrome decoder.

Bit positions are numbered 1..n. Parity bits sit at the powers of two
(1, 2, 4, 8, ...) and data bits fill the remaining positions. With this
layout, the syndrome of a received word is exactly the binary index of the
flipped bit, so single-bit errors are corrected by flipping that position.

Everything is vectorised with NumPy: a whole batch of blocks is encoded or
decoded with a single matrix multiplication (mod 2).
"""

import numpy as np


class HammingCode:
    def __init__(self, r: int):
        """Hamming code with r parity bits: n = 2^r - 1, k = n - r."""
        self.r = r
        self.n = 2 ** r - 1
        self.k = self.n - r

        positions = np.arange(1, self.n + 1)                 # 1..n
        is_parity = (positions & (positions - 1)) == 0       # powers of two
        self.parity_pos = np.where(is_parity)[0]             # 0-based indices
        self.data_pos = np.where(~is_parity)[0]

        # Parity-check matrix H (r x n): column j is the binary form of j+1.
        self.H = np.array([(positions >> i) & 1 for i in range(r)], dtype=np.uint8)

        # Generator matrix G (k x n): row i is the codeword of unit vector e_i.
        self.G = np.zeros((self.k, self.n), dtype=np.uint8)
        for i, dpos in enumerate(self.data_pos):
            self.G[i, dpos] = 1
            # parity bit at position 2^b covers every position with bit b set
            for b, ppos in enumerate(self.parity_pos):
                if (dpos + 1) >> b & 1:
                    self.G[i, ppos] = 1

        # Sanity check: every codeword satisfies H * c^T = 0 (mod 2).
        assert not ((self.G @ self.H.T) % 2).any()

    def encode(self, data: np.ndarray) -> np.ndarray:
        """data: (..., k) bits -> codewords: (..., n) bits."""
        return (data @ self.G % 2).astype(np.uint8)

    def syndrome(self, received: np.ndarray) -> np.ndarray:
        """Integer syndrome (0 = no error detected, else 1-based error position)."""
        s_bits = received @ self.H.T % 2                      # (..., r)
        weights = 1 << np.arange(self.r)
        return (s_bits * weights).sum(axis=-1)

    def decode(self, received: np.ndarray) -> np.ndarray:
        """Correct up to one error per block and return the k data bits."""
        single = received.ndim == 1                           # allow a single block too
        corrected = np.atleast_2d(received).copy()
        s = self.syndrome(corrected)
        idx = np.nonzero(s)                                   # blocks with a non-zero syndrome
        corrected[idx + (s[idx] - 1,)] ^= 1                   # flip the indicated position
        data = corrected[..., self.data_pos]
        return data[0] if single else data


HAMMING_7_4 = HammingCode(3)
HAMMING_15_11 = HammingCode(4)
