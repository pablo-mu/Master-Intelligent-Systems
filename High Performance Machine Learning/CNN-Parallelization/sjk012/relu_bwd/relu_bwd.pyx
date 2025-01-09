import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def relu_bwd_cython(dy, mask):
    shape = dy.shape
    cdef np.ndarray dx = np.zeros((np.prod(shape)), dtype=dy.dtype)

    relu_bwd_cython_inner(dx.reshape(-1), dy.reshape(-1), 
                                       mask.astype(np.int8).reshape(-1))
    return dx.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_bwd_cython_inner(np.ndarray[np.float32_t, ndim=1] dx,
                           np.ndarray[np.float32_t, ndim=1] dy,
                           np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i
    ###########################################################################
    # TODO: Implement the relu_backward operation using Cython code.          #
    # Parallelize the loop using prange.                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Define the number of elements in the array
    for i in prange(dx.shape[0], nogil=True):
        dx[i] = dy[i] if mask[i] else 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################        
