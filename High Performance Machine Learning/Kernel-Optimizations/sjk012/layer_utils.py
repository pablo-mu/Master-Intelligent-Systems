from .layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = fc_forward(x, w, b)
    out, relu_cache = relu_forward_cython(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def fc_relu_backward(dy, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward_cython(dy, relu_cache)
    dx, dw, db = fc_backward(da, fc_cache)
    return dx, dw, db

