"""
Minimal demonstration of Numba's just-in-time (JIT) compilation.

Numba (https://numba.pydata.org) compiles a Python function to optimised
machine code the first time it is called. The first call therefore pays a
one-off compilation cost; subsequent calls run the compiled version and are
much faster. This script times a function across both calls to make that
difference visible.
"""
import time

import numpy as np
from numba import jit


@jit(nopython=True)  # nopython mode gives the best performance (same as @njit)
def go_fast(a: np.ndarray) -> np.ndarray:
    """Add the trace of a square matrix to every element.

    The body deliberately mixes a Python loop, a NumPy function and
    broadcasting so that Numba has something non-trivial to compile.
    """
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


def main() -> None:
    x = np.arange(100).reshape(10, 10)

    # First call: includes the one-off JIT compilation cost.
    start = time.perf_counter()
    go_fast(x)
    print(f"Time elapsed (with compilation):    {time.perf_counter() - start:.8f}s")

    # Second call: the function is already compiled, so this is the true runtime.
    start = time.perf_counter()
    go_fast(x)
    print(f"Time elapsed (after compilation):   {time.perf_counter() - start:.8f}s")


if __name__ == "__main__":
    main()
