'''
 From the jupyter lab projects, develop a standalone python module that is able to execute the neural network in your computer. 
 It should report the network accuracy and its execution time.
 
'''

# Importing the necessary libraries
import numpy as np
import shutil
import time
import numpy as np
from icecream import ic
import os, textwrap
import sys


# Setup the environment 
notebook_path = ic(os.getcwd())
original_ld_library_path = ic(os.environ.get('LD_LIBRARY_PATH'))
# Use the pre-installed OpenBLAS and BLIS on the server
os.environ['LD_LIBRARY_PATH'] = os.path.join('/home/al437004/optb', 'blis', 'lib')
os.environ['LD_LIBRARY_PATH'] += ":"
os.environ['LD_LIBRARY_PATH'] += os.path.join('/home/al437004/optb', 'openblas', 'lib')
if original_ld_library_path is not None:
    os.environ['LD_LIBRARY_PATH'] += ":"
    os.environ['LD_LIBRARY_PATH'] += original_ld_library_path
_ = ic(os.environ['LD_LIBRARY_PATH'])


''' Uncomment this part of the code in order to execute it locally. 
# Setup the environment 
notebook_path = ic(os.getcwd())
original_ld_library_path = ic(os.environ.get('LD_LIBRARY_PATH'))
# Software and install_prefix as defined in 01_setup.ipynb
software_path = ic(os.path.join(notebook_path, 'sjk012', 'software'))
install_prefix = ic(os.path.join(software_path, 'opt'))

# Update the environment
os.environ['LD_LIBRARY_PATH'] = os.path.join(install_prefix, 'blis', 'lib')
os.environ['LD_LIBRARY_PATH'] += ":"
os.environ['LD_LIBRARY_PATH'] += os.path.join(install_prefix, 'openblas', 'lib')
if original_ld_library_path is not None:
    os.environ['LD_LIBRARY_PATH'] += ":"
    os.environ['LD_LIBRARY_PATH'] += original_ld_library_path
_ = ic(os.environ['LD_LIBRARY_PATH'])

# Install required packages
os.system("pip install numpy matplotlib threadpoolctl imageio cython pickleshare icecream")
'''


import pyximport
pyximport.install()

from sjk012.classifiers.cnn import ThreeLayerConvNet
from sjk012.data_utils import get_CIFAR10_data
from sjk012.solver import Solver

# Load the CIFAR-10 data
data = get_CIFAR10_data()

all_data = {
   'X_train': data['X_train'].astype(np.float32),
   'y_train': data['y_train'].astype(np.int8),
   'X_val': data['X_val'].astype(np.float32),
   'y_val': data['y_val'].astype(np.int8),
}
# Create a small net and solver
model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

# Measure execution time
start_time = time.time()

print("Start training")
solver = Solver(
    model,
    all_data,
    num_epochs=1,
    batch_size=50,
    update_rule='adam',
    optim_config={'learning_rate': 1e-3,},
    verbose=True,
    print_every=20
)
solver.train()


end_time = time.time()

# Report results
training_time = end_time - start_time
train_accuracy = solver.check_accuracy(data['X_train'].astype(np.float32), data['y_train'])
val_accuracy = solver.check_accuracy(data['X_val'].astype(np.float32), data['y_val'])

print(f"Training time: {training_time:.2f} seconds")
print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Validation accuracy: {val_accuracy:.2f}")