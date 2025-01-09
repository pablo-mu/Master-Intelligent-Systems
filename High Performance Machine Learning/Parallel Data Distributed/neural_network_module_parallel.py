'''
 Implementation using mpi4py in order to distribute parts of the data to different mpi processes
 and perform a data parallel version of the previous neural network
 (synchronizing the data between the different mpi processes whenever it is necessary).
'''


# Importing the necessary libraries
import numpy as np
import shutil
import time
from icecream import ic
import os, textwrap
import sys
from mpi4py import MPI
import pyximport

# Environment preparation
notebook_path = ic(os.getcwd())
original_ld_library_path = ic(os.environ.get('LD_LIBRARY_PATH'))

# The next block of lines should be commented to run it in patan.
software_path = ic(os.path.join(notebook_path, 'sjk012', 'software'))
install_prefix = ic(os.path.join(software_path, 'opt'))
os.environ['LD_LIBRARY_PATH'] = os.path.join(install_prefix, 'blis', 'lib')
os.environ['LD_LIBRARY_PATH'] += ":"
os.environ['LD_LIBRARY_PATH'] += os.path.join(install_prefix, 'openblas', 'lib')
if original_ld_library_path is not None:
    os.environ['LD_LIBRARY_PATH'] += ":"
    os.environ['LD_LIBRARY_PATH'] += original_ld_library_path
_ = ic(os.environ['LD_LIBRARY_PATH'])


pyximport.install(reload_support=True, pyimport=True) # Compile cython

try:
    from sjk012.classifiers.cnn import ThreeLayerConvNet
    from sjk012.data_utils import get_CIFAR10_data
    from sjk012.solver import Solver
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sys.exit(1)



# Initialize mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    print(f"MPI initialized with {size} processes")

''' This is another version in which, each process loads the data and then select its portion of the data
However this version is not used, because memory problems arise when loading the data in each process. (Working on WSL)
data = get_CIFAR10_data()

# Verificar que los datos se hayan cargado correctamente en cada proceso
if data is None:
    print(f"Rank {rank}: Data loading failed.")
else:
    print(f"Rank {rank}: Data loaded successfully.")

# Split the data among processes
num_train_per_process = data['X_train'].shape[0] // size
start_idx = rank * num_train_per_process
end_idx = (rank + 1) * num_train_per_process if rank != size - 1 else len(data['X_train'])

local_data = {
    'X_train': data['X_train'][start_idx:end_idx].astype(np.float32),
    'y_train': data['y_train'][start_idx:end_idx].astype(np.int8),
    'X_val': data['X_val'].astype(np.float32),
    'y_val': data['y_val'].astype(np.int8),
}

del data # Free up memory
'''


# Load the CIFAR-10 data
if rank == 0:
    print("Loading CIFAR-10 data...")
    data = get_CIFAR10_data()
    
    # Split the training data into chunks, one for each process
    X_train_chunks = np.array_split(data['X_train'], size)
    y_train_chunks = np.array_split(data['y_train'], size)
else:
    data = None
    X_train_chunks = None
    y_train_chunks = None

# Scatter the train data to all processes
train_data = comm.scatter(X_train_chunks, root=0)
train_labels = comm.scatter(y_train_chunks, root=0)


# Broadcast the validation data to all processes
if rank == 0:
    val_data = data['X_val']
    val_labels = data['y_val']
else:
    val_data = None
    val_labels = None

# Broadcast the validation data to all processes
val_data = comm.bcast(val_data, root=0)
val_labels = comm.bcast(val_labels, root=0)

# Debugging information
print(f"Rank {rank}: Training data shape: {train_data.shape}")
print(f"Rank {rank}: Validation data shape: {val_data.shape}")

# Create a dictionary with the local data
local_data = {
    'X_train': train_data.astype(np.float32),
    'y_train': train_labels.astype(np.int8),
    'X_val': val_data.astype(np.float32),
    'y_val': val_labels.astype(np.int8),
}

# Free up memory
del data

# Create a small net and solver
model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

# Broadcast one model parameters to all processes to ensure all processes start with the same model
model_params = comm.bcast(model.params, root=0)
model.params = model_params


# Measure execution time
start_time = time.time()


if rank == 0:
    print("Start training")
    
solver = Solver(
    model,
    local_data,
    num_epochs=2,
    batch_size=50,
    update_rule='adam',
    optim_config={'learning_rate': 1e-3,},
    verbose= True if rank == 0 else False,
    print_every=20
)

## MPI train version. 

def mpi_train(solver, comm, rank, size):
    num_train = solver.X_train.shape[0]
    iterations_per_epoch = max(num_train // solver.batch_size, 1)
    num_iterations = solver.num_epochs * iterations_per_epoch
    
    for t in range(num_iterations):
        solver._step()
        # Reduce all the gradients to synchronyze the computation
        for param in solver.model.params:
            param_data = solver.model.params[param]
            global_param_data = np.zeros_like(param_data)
            comm.Allreduce(param_data, global_param_data, op=MPI.SUM)
            solver.model.params[param] = global_param_data / size    

        # Maybe print training loss
        if solver.verbose and t % solver.print_every == 0:
            print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, solver.loss_history[-1])
                )
        # At the end of every epoch, increment the epoch counter and decay the learning rate.
        epoch_end = (t + 1) % iterations_per_epoch == 0
        if epoch_end:
            solver.epoch += 1
            for k in solver.optim_configs:
                solver.optim_configs[k]['learning_rate'] *= solver.lr_decay
        
        # Check train and val accuracy on the first iteration, 
        # the last iteration, and the end of each epoch.
        # This is made in each process and then obtain the mean. 
        first_it = t == 0
        last_it = t == num_iterations - 1
        if first_it or last_it or epoch_end:
                train_acc = solver.check_accuracy(
                    solver.X_train, solver.y_train, num_samples=solver.num_train_samples
                )
                val_acc = solver.check_accuracy(
                    solver.X_val, solver.y_val, num_samples=solver.num_val_samples
                )

                global_train_acc = comm.reduce(train_acc, op=MPI.SUM, root=0)
                global_val_acc = comm.reduce(val_acc, op=MPI.SUM, root=0)
        
                if rank == 0:
                    global_train_acc /= size
                    global_val_acc /= size
                    solver.train_acc_history.append(global_train_acc)
                    solver.val_acc_history.append(global_val_acc)
                    solver._save_checkpoint()
        
                if solver.verbose and rank == 0:
                    print(f"(Epoch {solver.epoch} / {solver.num_epochs}) "
                      f"train acc: {global_train_acc}; val_acc: {global_val_acc}")

                if rank == 0 and global_val_acc > solver.best_val_acc:
                    solver.best_val_acc = global_val_acc
                    solver.best_params = {}
                    for k, v in solver.model.params.items():
                        solver.best_params[k] = v.copy()
        
        solver.best_params = comm.bcast(solver.best_params, root=0)
        solver.model.params = solver.best_params
        comm.Barrier()


mpi_train(solver, comm, rank, size)

        


end_time = time.time()

# Report results
training_time = end_time - start_time
local_train_accuracy = solver.check_accuracy(local_data['X_train'], local_data['y_train'])
local_val_accuracy = solver.check_accuracy(local_data['X_val'], local_data['y_val'])

# Gather results from all processes
train_accuracies = comm.gather(local_train_accuracy, root=0)
val_accuracies = comm.gather(local_val_accuracy, root=0)

if rank == 0:
    avg_train_accuracy = np.mean(train_accuracies)
    avg_val_accuracy = np.mean(val_accuracies)

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {avg_train_accuracy:.2f}")
    print(f"Validation accuracy: {avg_val_accuracy:.2f}")

# Finalize MPI
MPI.Finalize()