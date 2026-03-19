import numpy as np
import pytest

pytestmark = pytest.mark.integration

num = 10000000


if __name__ == "__main__":
    from numba import jit, cuda  # noqa: F401
    from timeit import default_timer as timer

    # normal function to run on cpu
    def func_cpu(a):
        for i in range(num):
            a[i] += 1

    # function optimized to run on gpu
    @jit(target_backend="cuda")
    def func_gpu(a):
        for i in range(num):
            a[i] += 1

    a = np.ones(num, dtype=np.float64)
    start = timer()
    func_cpu(a)
    print("Without GPU:", timer() - start)
    start = timer()
    func_gpu(a)
    print("With GPU:", timer() - start)
