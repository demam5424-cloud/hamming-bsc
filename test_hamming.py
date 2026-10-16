"""
Exhaustive check of the Hamming(7,4) encoder/decoder (Task 1):
all 16 data words, with no error and with every possible single-bit error.
Also checks Hamming(15,11) on all 2048 data words.

Run:  python test_hamming.py
"""

import itertools
import numpy as np
from hamming import HAMMING_7_4, HAMMING_15_11


def check_code(code, verbose=False):
    words = np.array(list(itertools.product([0, 1], repeat=code.k)), dtype=np.uint8)
    codewords = code.encode(words)

    # 1) No error: decoding gives back the original data.
    assert np.array_equal(code.decode(codewords), words)

    # 2) Every single-bit error in every codeword is corrected.
    n_tests = 0
    for pos in range(code.n):
        received = codewords.copy()
        received[:, pos] ^= 1
        assert np.array_equal(code.syndrome(received), np.full(len(words), pos + 1))
        assert np.array_equal(code.decode(received), words)
        n_tests += len(words)

    print(f"Hamming({code.n},{code.k}): {len(words)} words OK without error, "
          f"{n_tests} single-error cases all corrected.")

    if verbose:
        print("\n data  -> codeword (positions 1..7 = p1 p2 d1 p3 d2 d3 d4)")
        for w, c in zip(words, codewords):
            print(" ", "".join(map(str, w)), "->", "".join(map(str, c)))


if __name__ == "__main__":
    check_code(HAMMING_7_4, verbose=True)
    check_code(HAMMING_15_11)
    print("\nAll tests passed.")
