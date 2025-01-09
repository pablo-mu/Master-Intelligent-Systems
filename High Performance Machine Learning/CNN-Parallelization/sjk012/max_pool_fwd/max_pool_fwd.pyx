import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def max_pool_fwd_cython(x, filter_height, filter_width, padding, stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int HH = (H + 2 * padding - filter_height) / stride + 1
    cdef int WW = (W + 2 * padding - filter_width) / stride + 1

    cdef np.ndarray y = np.zeros((N, C, HH, WW), dtype=x.dtype)
    cdef np.ndarray idx_max = np.zeros((N, C, HH, WW), dtype=np.int32)

    max_pool_fwd_cython_inner(y, x, idx_max, N, H, W, C, HH, WW, 
                              filter_height, filter_width, padding, stride)
    return y, idx_max


@cython.boundscheck(False)
@cython.wraparound(False)
cdef max_pool_fwd_cython_inner(np.ndarray[np.float32_t, ndim=4] y,
                               np.ndarray[np.float32_t, ndim=4] x,
                               np.ndarray[np.int32_t, ndim=4] idx_max,
                               int N, int H, int W, int C, int HH, int WW,
                               int filter_height, int filter_width, int padding, int stride):
    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef np.float32_t maxval, minval, val
    minval = np.finfo(np.float32).min

    for nn in prange(N, nogil=True):
        for cc in range(C):
            for xx in range(HH):
                for yy in range(WW):
                    maxval, idx_maxval = minval, 0
                    for ii in range(filter_height):
                        x_x = stride * xx + ii - padding
                        if 0 <= x_x < H:
                            for jj in range(filter_width):
                                x_y = stride * yy + jj - padding
                                if 0 <= x_y < W:
                                    val = x[nn, cc, x_x, x_y]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * filter_width + jj
                    y[nn, cc, xx, yy], idx_max[nn, cc, xx, yy] = maxval, idx_maxval
