import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def col2im_cython(cols, N, C, H, W, filter_height, filter_width, padding, stride):
    
    cdef np.ndarray x = np.zeros((N, C, H, W), dtype=cols.dtype)
    
    cdef int HH = (H + 2 * padding - filter_height) / stride + 1
    cdef int WW = (W + 2 * padding - filter_width) / stride + 1

    col2im_cython_inner(cols, x, N, C, H, W, HH, WW, filter_height, filter_width, padding, stride)
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef col2im_cython_inner(np.ndarray[np.float32_t, ndim=2] cols,
                         np.ndarray[np.float32_t, ndim=4] x,
                         int N, int C, int H, int W, int HH, int WW,
                         int filter_height, int filter_width, int padding, int stride):
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(C, nogil=True):
        for ii in range(filter_height):
            for jj in range(filter_width):
                row = cc * filter_height * filter_width + ii * filter_width + jj
                for nn in range(N):
                    for xx in range(HH):
                        x_x = stride * xx + ii - padding
                        if 0 <= x_x < H:
                            for yy in range(WW):
                                x_y = stride * yy + jj - padding
                                if 0 <= x_y < W:
                                    col = nn * HH * WW + xx * WW + yy
                                    x[nn, cc, x_x, x_y] += cols[row, col]