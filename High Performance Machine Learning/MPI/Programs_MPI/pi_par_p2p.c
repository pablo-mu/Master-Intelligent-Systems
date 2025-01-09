/*
Complete the file “pi_par_p2p.c”, so that each process accumulates all its calculations 
on a local variable and the result is obtained accumulating all these values.
First, the variable num_steps must be sent from process 0 to the other processes. 
Later, all the processes send the local variable sum to process 0, which receives all 
the values and accumulates them pi. All communications are done using point-to-point 
communications.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int i;
  double t1, t2, t_seq, t_par, sp, ep;

  int rank, size, rc;
  MPI_Status st;

  // MPI Initialization
  rc = MPI_Init (&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);// Aborting the execution
  }
  // Get the rank of the process
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  // Get the number of processes
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  // Adjust the number of rectangules
  int num_steps = 100000;
  if (rank == 0) {
    if (argc == 2) {
      num_steps = atoi(argv[1]);
    } else if (argc > 2) {
      printf("Wrong number of parameters\n");
      printf("mpirun -np ?? ./a.out [ num_steps ]\n");
      MPI_Abort(MPI_COMM_WORLD, rc);// Aborting the execution
    }
  }
  // Sending num_steps from process 0 to the other processes
  if (rank == 0) {
    for (i = 1; i < size; i++) {
      MPI_Send(&num_steps, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&num_steps, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
  }
  printf ("num_steps(%d) = %d\n", rank, num_steps);

/*************************************/
/******** Computation of pi **********/
/*************************************/

  double step = 1.0 / (double) num_steps;  
  double x, pi, sum = 0.0;

  //
  // Sequential implementation
  //
  if (rank == 0) {
    // Computation without getting time
    pi = 0.0;
    step = 1.0 / (double) num_steps;  
    for (i=0; i<num_steps; i++){
      x = (i+0.5)*step;
      sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    // Computation getting time
    t1 = MPI_Wtime( );
    sum = 0.0;
    step = 1.0 / (double) num_steps;  
    for (i=0; i<num_steps; i++){
      x = (i+0.5)*step;
      sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    t2 = MPI_Wtime( );
    t_seq = (t2-t1);
  
    printf(" pi_seq = %20.15f\n", pi);
    printf(" time_seq = %20.15f\n", t_seq);
  }

  //
  // Parallel implementation
  //

  // Getting the first tick
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime( );

  // Local computation of pi
  sum = 0.0;
  step = 1.0 / (double) num_steps;
  for (i = rank; i < num_steps; i += size) {
    x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }
  sum = step * sum;

  // Sending local computations to process 0,
  // which accumulates them and obtains pi
  if (rank == 0) {
    pi = sum;
    for (i = 1; i < size; i++) {
      MPI_Recv(&sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &st);
      pi += sum;
    }
  } else {
    MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  // Getting the final tick and calculating performance indices
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime( );
  t_par = (t2 - t1);
  sp = t_seq / t_par;
  ep = sp / size;

  if (rank == 0) {
    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
  }

  // MPI Finalization
  MPI_Finalize();
}
