import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def matmul_tiled_cython(a, b, c, int block_size):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    matmul_tiled_cython_inner(a, b, c, m, n, k, block_size)
    return c

@cython.boundscheck(False)
@cython.wraparound(False)
cdef matmul_tiled_cython_inner(np.ndarray[np.float32_t, ndim=2] a,
                               np.ndarray[np.float32_t, ndim=2] b,
                               np.ndarray[np.float32_t, ndim=2] c,
                               int m, int n, int k,
                               int block_size):
    
    cdef int m_, n_, k_, nc_, kc_, i, j, l, m_upper, n_upper, k_upper
    cdef float temp

    ###########################################################################
    # TODO: Implement the tiled matmul operation using Cython code.           #
    # Parallelize the loop via prange.                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for m_ in prange(0, m, block_size, nogil = True):
        for n_ in prange(0,n,block_size):
            for k_ in prange(0,k, block_size):
                m_upper = min(m_+block_size, m)
                n_upper = min(n_ + block_size, n)
                k_upper = min(k_ + block_size, k)
                for i in range(m_ , m_upper):
                    for j in range(n_, n_upper):
                        temp = 0
                        for l in range(k_, k_upper):
                            temp += a[i, l] * b[l, j]
                        c[i, j] += temp
    return c
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
                
                        