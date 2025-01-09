#!/usr/bin/env bash

# Bash options
set -o errexit
set -o nounset


echo "This is work in progress!"
echo "Currently, it does not work as expected"
exit 1

# Script dir
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PREFIX_DIR=$(readlink -f "${SCRIPT_DIR}/../software/opt/blis")

error() {
    echo $* >&2
    exit 1
}

[ -d ${PREFIX_DIR} ] || error "Directory '${PREFIX_DIR}' not found. Have you installed BLIS? Aborting!"

echo "Creating '~/.numpy-site.cfg'..."
cat > ~/.numpy-site.cfg <<EOF
[blis]
libraries = blis
library_dirs = ${PREFIX_DIR}/lib
include_dirs = ${PREFIX_DIR}/include/blis
runtime_library_dirs = ${PREFIX_DIR}/lib
EOF

echo "Installing numpy from source..."
export NPY_BLAS_LIBS="blis"
export NPY_CBLAS_LIBS=""
export NPY_BLAS_ORDER="blis"
export OPT="-lgomp"
export LD_LIBRARY_PATH="${PREFIX_DIR}/lib"
pip install --no-binary=':all:' --force-reinstall --no-cache-dir numpy


exit 0

cat <<EOF

Proposal:

In the last cell of 01_setup.ipynb
--------------------------
## Linking NumPy with BLIS

When numpy is binary installed using pip, the numpy library is linked against a OpenBLAS library that is provided by the numpy package itself.

After performing the *Parallel Performance Evaluation* you should see that there is a big difference between the results obtained with the NumPy and Cython implementations.

After linking Numpy with BLIS, you should rerun the experiments and see what happens.

**Therefore, the next step should be done later (not the first time)**
--------------------------
# Uncomment the next line to link NumPy with BLIS

#!bash sjk012/bin/install_numpy_with_blis.sh
--------------------------
import numpy as np
np.show_config()
--------------------------


In 02_assignment2_01.pynb (after Parallel Performance Evaluation cells)

--------------------------
# Parallel Performance Evaluation (with NumPy linked against BLIS)

Go back to `01_setup.py` and performe the step *
In this section, we will evaluate the performance of the numpy and im2col-based convolutions using the parallel version.

### Task Description

1. **Plot Execution Times**: Plot the execution times of both numpy and im2col-based convolutions, varying the number of threads and adjusting parameters such as batch size, number of filters, and input channels.

2. **Observations**: Analyze the execution times of each convolution version under different scenarios, considering variations in the batch size, image size, filter size, number of filters, and number of channels.

### Observations

- **Scalability with Threads**: Determine which version scales better with respect to the number of threads and other parameters.

- **Efficiency Comparison**: Assess the efficiency of each convolution version under various scenarios.

*Write your observations here:*
--------------------------

EOF