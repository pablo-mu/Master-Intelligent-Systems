from builtins import range
from .operations import matmul
import numpy as np

def fc_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    x_shape = x.shape
    x = x.reshape(x_shape[0], -1)
    ###########################################################################
    # TODO: Implement the affine forward pass using the matmul function       #
    # declared in operations.py.  Store the result in out.                    #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = matmul(x, w) + b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, x_shape)
    return out, cache


def fc_backward(dy, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dy: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b, x_shape = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    # For the backward pass, we need to compute gradients dx, dw, and db      #
    # with respect to inputs x, weights w, and biases b, respectively.        #
    # Here are the steps for the backward pass:                               #
    # 1. Reshape the input x to 2D, keeping the batch size unchanged.         #
    #    This allows us to perform matrix multiplication efficiently.         #
    # 2. Compute the gradient of the input with respect to the loss (dx).     #
    #    This is done by multiplying the upstream gradient (dy) by the        #
    #    transpose of the weight matrix (w^T). The resulting dx has the same  #
    #    shape as the original input x.                                       #
    # 3. Compute the gradient of the weights with respect to the loss (dw).   #
    #    This is done by multiplying the transpose of the reshaped input      #
    #    (x_reshaped^T) by the upstream gradient (dy). The resulting dw       #
    #    has the same shape as the weight matrix w.                           #
    # 4. Compute the gradient of the biases with respect to the loss (db).    #
    #    This is simply the sum of the upstream gradient (dy) along each      #
    #    dimension, representing the contribution of each sample in the batch #
    #    to the bias gradient.                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Reshape x into 2D, keeping the batch size unchanged
    x = x.reshape(x_shape[0], -1)
    
    #Compute the gradient of the input with respect to the loss
    dx = matmul(dy, w.T).reshape(x_shape)
    
    # Compute the gradient of the weights with respect to the loss
    dw = matmul(x.T, dy)
    
    # Compute the gradient of the biases with respect to the loss
    db = np.sum(dy, axis=0)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
    

def relu_forward_numpy(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass using numpy maximum function      #
    # and the numpy vector comparison operator                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # ReLU forward pass
    y = x*(x>0)
    mask = x
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return y, mask


def relu_forward_cython(x):
    from sjk012.relu_fwd.relu_fwd import relu_fwd_cython
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass calling the relu function         #
    # available in operations.py                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y, cache = relu_fwd_cython(x)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return y, cache


def relu_backward_numpy(dy, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, mask = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dy*(mask>0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
    

def relu_backward_cython(dy, cache):
    from sjk012.relu_bwd.relu_bwd import relu_bwd_cython
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, mask = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx = relu_bwd_cython(dy, mask)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    N = len(y) # number of samples

    P = np.exp(x - x.max(axis=1, keepdims=True)) # numerically stable exponents
    P /= P.sum(axis=1, keepdims=True)            # row-wise probabilities (softmax)

    loss = -np.log(P[range(N), y]).sum() / N # sum cross entropies as loss

    P[range(N), y] -= 1
    dx = P / N

    return loss, dx
    

def conv_forward_numpy(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pad = conv_param['pad']
    stride = conv_param['stride']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Calculate the output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Initialize the output
    out = np.zeros((N, F, H_out, W_out))

    # Perform the convolution
    for n in range(N):
      for f in range(F):
        for i in range(H_out):
          for j in range(W_out):
            # Extract the receptive field
            receptive_field = x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            
            # Perform the convolution
            out[n, f, i, j] = np.sum(receptive_field * w[f]) + b[f]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_cython(x, w, b, conv_param):
    from sjk012.im2col.im2col import im2col_cython
    ###########################################################################
    # TODO: Implement the forward pass for a convolutional layer using NumPy. #
    # Use the im2col_numpy function to convert the input tensor into columns, #
    # perform the convolution operation using matrix multiplication,          #
    # and add the bias.                                                       #
    # You need to use the parameters from conv_param to determine the stride  #
    # and padding.                                                            #
    # Return the output tensor.                                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Extract parameters and get dimensions
    stride = conv_param['stride']
    padding = conv_param['pad']
    N, C, H, W = x.shape
    F, _, filter_height, filter_width = w.shape

    # Compute output dimensions
    H_out = 1 + (H + 2 * padding - filter_height) // stride
    W_out = 1 + (W + 2 * padding - filter_width) // stride

    # Reshape the weights to a 2D array (F, C * filter_height * filter_width)
    w_reshaped = w.reshape(F, -1)

    # Perform im2col transformation
    x_cols = im2col_cython(x, filter_height, filter_width, padding, stride)

    # Perform convolution by matrix multiplication
    out = np.dot(w_reshaped, x_cols) + b.reshape(-1, 1)

    # Reshape the output to the correct shape (N, F, H_out, W_out)
    
    out = out.reshape(F, N, H_out, W_out).transpose(1, 0, 2, 3)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ########################################################################### 
    
    cache = (x.shape, x_cols, w, b, conv_param)
    return out, cache


def conv_backward_numpy(dy, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dy: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Extract the cache and get the dimensions
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dy.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    # Initialize the gradients
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Pad the input and its respective gradient
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    # Perform the backward pass
    for n in range(N):
      for f in range(F):
        for i in range(H_out):
          for j in range(W_out):
            # Extract the receptive field
            receptive_field = x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            
            # Compute the gradients
            dw[f] += receptive_field * dy[n, f, i, j]
            db[f] += dy[n, f, i, j]
            dx_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f] * dy[n, f, i, j]
    
    # Remove padding from dx
    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dw, db


def conv_backward_cython(dy, cache):
    from sjk012.col2im.col2im import col2im_cython
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dy: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None    
    ###########################################################################
    # TODO: Implement the backward pass for a convolutional layer using NumPy.
    # You should compute the gradients of the input data (dx), the weights (dw),
    # and the biases (db). The function should return dx, dw, and db.
    # Use the im2col_numpy function to convert the input tensor into columns.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
  
    # Extract the cache and get the dimensions
    x_shape, x_cols, w, b, conv_param = cache
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    # Transpose and reshape output gradient
    dy_reshaped = dy.transpose(1,0,2,3).reshape(F,-1)
    
    # Compute gradient with respect to the filters.
    dw = np.dot(dy_reshaped, x_cols.T).reshape(w.shape)
    
    # Compute gradient with respect to the biases
    db = np.sum(dy, axis=(0, 2, 3))
    
    # Compute gradient with respect to the input data
    dx_cols = np.dot(w.reshape(F, -1).T, dy_reshaped)
    
    # Convert dx_cols gradient to image format

    dx = col2im_cython(dx_cols, x_shape[0], x_shape[1], x_shape[2], x_shape[3], HH, WW, pad, stride)

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dw, db


def max_pool_forward_numpy(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #Get the dimensions
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
    # Calculate the output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    # Initialize the output
    out = np.zeros((N, C, H_out, W_out))
    
    # Perform the max pooling
    for n in range(N):
      for c in range(C):
        for i in range(H_out):
          hs = i * stride
          for j in range(W_out):
            ws = j * stride
            window = x[n, c, hs:hs+pool_height, ws:ws+pool_width]
            out[n,c,i,j] = np.max(window)
            
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_forward_cython(x, pool_param):
    from sjk012.max_pool_fwd.max_pool_fwd import max_pool_fwd_cython
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    pool_height, pool_width, stride, padding = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'], pool_param['padding'] 
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    
    out, idx_max = max_pool_fwd_cython(x, pool_height, pool_width,padding, stride)
    out = out.reshape(N, C, H_out, W_out)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (idx_max, x.shape, pool_param)
    return out, cache


def max_pool_backward_numpy(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Extract the cache and pool parameters
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # Get the dimensions
    N, C, H_out, W_out = dout.shape
    _, _, H, W = x.shape
    
    # Initialize the gradient
    dx = np.zeros_like(x)
    
    # Perform the backward pass
    for n in range(N):
      for c in range(C):
        for i in range(H_out):
          hs = i * stride
          for j in range(W_out):
            ws = j * stride
            window = x[n, c, hs:hs+pool_height, ws:ws+pool_width]
            mask = window == np.max(window)
            dx[n, c, hs:hs+pool_height, ws:ws+pool_width] += mask * dout[n, c, i, j]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def max_pool_backward_cython(dy, cache):
    from sjk012.max_pool_bwd.max_pool_bwd import max_pool_bwd_cython
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Extract the cache and pool parameters
    idx_max, x_shape, pool_param = cache
    N, C, H, W = x_shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    padding = pool_param['padding']
    
    dx = max_pool_bwd_cython(dy, idx_max, N, H, W, C, pool_height, pool_width, padding, stride)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        pass
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Compute the sample mean and variance
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        #Normalize the data
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        
        #scale and shift the normalized data
        out = gamma * x_normalized + beta
        
        # update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        # Store the intermediate values in the cache
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
        
        #update bn_param 
        bn_param["running_mean"] = running_mean
        bn_param["running_var"] = running_var
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        pass
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Normalize the data using the running mean and variance
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        
        # Scale and shift
        out = gamma * x_normalized + beta
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Unpack the cache
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    
    # Get the dimensions
    N, D = x.shape
    
    # Compute the gradients
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    # Compute the gradient of the input with respect to the loss
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - sample_mean) * -0.5 * (sample_var + eps)**-1.5, axis=0)
    dmean = np.sum(dx_norm * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    dx = dx_norm / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / N + dmean / N
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Unpack the Cache
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    
    # Get the dimensions
    N, D = x.shape
    
    # Compute the gradients
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    # Compute the gradient of the input with respect to the loss
    dx = (1. / N) * gamma * (sample_var + eps)**-0.5 * (N * dout - np.sum(dout, axis=0) - (x - sample_mean) * (sample_var + eps)**-1.0 * np.sum(dout * (x - sample_mean), axis=0))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        pass
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #Create a mask of the same shape as x
        mask = (np.random.rand(*x.shape) < p) / p
        
        #Apply the mask to the input
        out = x * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        pass
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        pass
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    
    # Reshape x to (N*H*W, C) and apply batchnorm
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # Reshape the output back to the original shape
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    
    # Reshape dout to (N*H*W, C) and apply batchnorm backward
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    
    # Reshape the gradients back to the original shape
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W = x.shape
    x_group = x.reshape(N,G,C//G,H,W)
    
    # Compute the mean and variance
    mean = np.mean(x_group, axis=(2,3,4), keepdims=True)
    var = np.var(x_group, axis=(2,3,4), keepdims=True)
    
    # Normalize the data
    x_group_normalized = (x_group - mean) / np.sqrt(var + eps)
    
    # Reshape the normalized data
    x_normalized = x_group_normalized.reshape(N, C, H, W)
    
    # Scale and shift
    out = gamma * x_normalized + beta
    
    # Store the intermediate values in the cache
    cache = (x, x_normalized, mean, var, gamma, beta, G, eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_normalized, G, gamma, beta, mean, var, eps = cache
    N, C, H, W = dout.shape

    # Reshape dout and x_normalized to shape (N*G, C//G*H*W)
    dout_reshaped = dout.reshape(N*G, -1)
    x_normalized_reshaped = x_normalized.reshape(N*G, -1)

    # Calculate dbeta and dgamma
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)

    # Calculate dx
    dx_normalized = dout_reshaped * gamma.reshape(1, C, 1, 1)
    dx_grouped = dx_normalized.reshape(N, G, C // G, H, W)
    x_grouped = x_normalized.reshape(N, G, C // G, H, W)
    std_dev = np.sqrt(var + eps)

    dvar = np.sum(dx_grouped * (x_grouped - mean) * -0.5 * std_dev**-3, axis=(2, 3, 4), keepdims=True)
    dmean = np.sum(dx_grouped * -1 / std_dev, axis=(2, 3, 4), keepdims=True) + dvar * np.mean(-2 * (x_grouped - mean), axis=(2, 3, 4), keepdims=True)

    dx_grouped += (2.0 / (C // G * H * W) * dvar * (x_grouped - mean) + dmean / (C // G * H * W))
    dx = dx_grouped.reshape(N, C, H, W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
