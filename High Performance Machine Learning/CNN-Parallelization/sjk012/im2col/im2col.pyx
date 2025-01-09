import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def im2col_cython(x, filter_height, filter_width, padding, stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int HH = (H + 2 * padding - filter_height) / stride + 1
    cdef int WW = (W + 2 * padding - filter_width) / stride + 1

    cdef np.ndarray[np.float32_t, ndim=2] cols = np.zeros(
            (C * filter_height * filter_width, N * HH * WW),
            dtype=x.dtype)
    
    im2col_cython_inner(cols, x, N, C, H, W, HH, WW,
                        filter_height, filter_width, padding, stride)
    return cols


@cython.boundscheck(False)
@cython.wraparound(False)
cdef im2col_cython_inner(np.ndarray[np.float32_t, ndim=2] cols,
                                    np.ndarray[np.float32_t, ndim=4] x,
                                    int N, int C, int H, int W, int HH, int WW,
                                    int filter_height, int filter_width, 
                                    int padding, int stride):
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    ###########################################################################
    # TODO: Parallel populate the cols matrix from the correspondent x values #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for nn in prange(N, nogil=True):
        for cc in range(C):
            for ii in range(filter_height):
                for jj in range(filter_width):
                    row = cc * filter_height * filter_width + ii * filter_width + jj
                    for yy in range(HH):
                        for xx in range(WW):
                            x_x = xx * stride + jj - padding
                            x_y = yy * stride + ii - padding
                            if x_x >= 0 and x_x < W and x_y >= 0 and x_y < H:
                                col = yy * WW + xx
                                cols[row, nn * HH * WW + col] = x[nn, cc, x_y, x_x]
                            else:
                                cols[row, nn * HH * WW + col] = 0
            
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                                    