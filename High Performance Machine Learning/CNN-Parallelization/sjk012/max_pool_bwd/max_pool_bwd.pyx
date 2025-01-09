import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def max_pool_bwd_cython(y, idx_max, N, H, W, C, 
                        filter_height, filter_width, padding, stride):
    cdef int HH = (H + 2 * padding - filter_height) / stride + 1
    cdef int WW = (W + 2 * padding - filter_width) / stride + 1

    cdef np.ndarray x = np.zeros((N, C, H, W), dtype=y.dtype)

    max_pool_bwd_cython_inner(y, x, idx_max, N, H, W, C, HH, WW, 
                              filter_height, filter_width, 
                              padding, stride)
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_bwd_cython_inner(np.ndarray[np.float32_t, ndim=4] y,
                                   np.ndarray[np.float32_t, ndim=4] x,
                                   np.ndarray[np.int32_t, ndim=4] idx_max,
                                   int N, int H, int W, int C, int HH, int WW,
                                   int filter_height, int filter_width, 
                                   int padding, int stride):
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(N, nogil=True):
        for cc in range(C):
            for xx in range(HH):
                for yy in range(WW):
                    idx_maxval = idx_max[nn, cc, xx, yy]
                    ii, jj = idx_maxval // filter_height, idx_maxval % filter_height
                    x_x = stride * xx + ii - padding
                    x_y = stride * yy + jj - padding
                    if 0 <= x_x < H and 0 <= x_y < W:
                        x[nn, cc, x_x, x_y] += y[nn, cc, xx, yy]
