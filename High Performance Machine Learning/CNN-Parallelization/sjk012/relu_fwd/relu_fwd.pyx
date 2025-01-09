import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def relu_fwd_cython(x):
    shape = x.shape
    cdef np.ndarray max = np.zeros((np.prod(shape)), dtype=x.dtype)
    cdef np.ndarray mask = np.zeros((np.prod(shape)), dtype=np.int8)

    relu_fwd_cython_inner_float32(x.reshape(-1), max, mask)
    
    return max.reshape(shape), mask.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_fwd_cython_inner_float32(np.ndarray[np.float32_t, ndim=1] x,
                                   np.ndarray[np.float32_t, ndim=1] max,
                                   np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i

    ###########################################################################
    # TODO: Implement the ReLU activation using Cython code.                  #
    # Parallelize the loop using prange.                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################        
