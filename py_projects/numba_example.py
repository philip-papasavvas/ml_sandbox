"""
Created on: 18 Feb 22
Created by: Philip P

Example of using numba package (https://numba.pydata.org/numba-doc/latest/user/5minguide.html)
"""

from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # nopython mode for best performance, same as @njit
def go_fast(a): # fn is compiled to mahcine code when called first time
    trace = 0.0
    for i in range(a.shape[0]):     # loops
        trace += np.tanh(a[i, i])   # NumPy functions
    return a + trace                # numba broadcasting

print(go_fast(x))

# performance with compilation time
import time
start_t = time.time()
go_fast(x)
end_t = time.time()
print(f"Time elapsed with compilation = {end_t-start_t : .8f}s")

# now function has been compiled
start_t2 = time.time()
go_fast(x)
end_t2 = time.time()
print(f"Time elapsed after compilation = {end_t2-start_t2 : .8f}s")