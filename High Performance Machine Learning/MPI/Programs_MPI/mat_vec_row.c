/*
Complete the file “mat_vec_row.c”, so that all processes work together to compute a 
matrix-vector product. 
First, process 0 must distribute by block of rows the matrix and vectors. Then, the local 
computations have to be done, but this requires a complete copy of vector x, so 
communication is neccessary. Finally, the solution must be stored in process 0. All 
communications are done using collective communications.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int i, j;
  double t1, t2, t_seq, t_dis, t_par, t_rec, sp, ep;

  int rank, size, rc;
  MPI_Status st;

  // MPI Initialization
  rc = MPI_Init (&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    // Aborting the execution
  }
  // Get the rank of the process
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  // Get the number of processes
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  // Adjust the matrix dimensions
  int M = 100, N = 100, ML = 0, NL = 0;
  if (rank == 0) {
    if (argc == 2) {
      M = atoi(argv[1]);
    } else if (argc == 3) {
      M = atoi(argv[1]);
      N = atoi(argv[2]);
    } else if (argc > 3) {
      printf("Wrong number of parameters\n");
      printf("mpirun -np ?? ./a.out [ M [ N ] ]\n");
      exit(-1);
    }
    if (((M % size) > 0) || ((N % size) > 0)) {
      printf ("The number of rows (%d) and columns (%d) have ", M, N);
      printf ("to be multiple of number of processors (%d)\n", size);
      MPI_Abort(MPI_COMM_WORLD, rc);// Aborting the execution
    }
  }
  // Sending M and N from process 0 to the other processes
  // All the communications should done using collective communications
  int v[2] ={M,N};
  if (rank == 0) {
    MPI_Bcast(v, 2, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(v, 2, MPI_INT, 0, MPI_COMM_WORLD);
    M = v[0];
    N = v[1];
  }
  ML = M / size;
  NL = N / size;
  printf ("M(%d) = %d , N(%d) = %d , ML(%d) = %d , NL(%d) = %d\n", 
          rank, M, rank, N, rank, ML, rank, NL);

/*************************************/
/****** Computation of mat_mult ******/
/*************************************/

  double A[M][N], x[N], y[M], sum, sum_seq, sum_par;;

  if (rank == 0) {
    for (i=0; i<M; i++) {
      y[i] = 0;
      for (j=0; j<N; j++) {
        A[i][j] = ( ( rand( ) % 10) * 1.0 ) / 
                    ( ( rand( ) % 1000 ) + 1 );
      }
    }
    for (j=0; j<N; j++) {
      x[j] = ( ( rand( ) % 10) * 1.0 ) / 
               ( ( rand( ) % 1000 ) + 1 );
    }
  }

  //
  // Sequential implementation
  //
  if (rank == 0) {
    t1 = MPI_Wtime( );
    for (i=0; i<M; i++) {
      y[i] = 0;
      for (j=0; j<N; j++) {
        y[i] += A[i][j] * x[j];
      }
    }
    t2 = MPI_Wtime( );
    t_seq = (t2-t1);

    sum_seq = 0.0;
    for (i=0; i<M; i++) {
      sum_seq += y[i];
    }
    printf(" sum_seq = %20.15f\n", sum_seq);
    printf(" time_seq = %20.15f\n", t_seq);
  }

  //
  // Parallel implementation
  //

  double AL[ML][N], xG[N], xL[NL], yG[M], yL[ML];
  
  // Getting the first tick
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime( );

  // Distributing A to Al, x to xL, y to YL
  if (rank == 0) {
    MPI_Scatter(A, ML*N, MPI_DOUBLE, AL, ML*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(x, NL, MPI_DOUBLE, xL, NL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, ML, MPI_DOUBLE, yL, ML, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatter(NULL, 0, MPI_DOUBLE, AL, ML*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, 0, MPI_DOUBLE, xL, NL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, 0, MPI_DOUBLE, yL, ML, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Getting the final tick and calculating the distribution time
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t_dis = t2-t1;

  // Getting the first tick
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime( );

  // Gathering xL to xG on all processes
  MPI_Allgather(xL, NL, MPI_DOUBLE, xG, NL, MPI_DOUBLE, MPI_COMM_WORLD);
  // Computing the local product
  for (i = 0; i < ML; i++) {
    yL[i] = 0;
    for (j = 0; j < N; j++) {
      yL[i] += AL[i][j] * xG[j];
    }
  }

  // Getting the final tick and calculating performance indices
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime( );
  t_par = (t2 - t1);
  sp = t_seq / t_par;
  ep = sp / size;

  // Getting the first tick
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime( );

  // Gathering yL to yG on processor 0
  MPI_Gather(yL, ML, MPI_DOUBLE, yG, ML, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Getting the final tick and calculating the joining time
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t_rec = t2-t1;

  if (rank == 0) {
    sum_par = 0.0;
    for (i=0; i<M; i++) {
      sum_par += yG[i];
    }
    double sp_glb = t_seq / (t_dis + t_par + t_rec);
    printf(" sum_par = %20.15f , diff = %20.15e\n", sum_par, sum_seq-sum_par);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
    printf(" time_dis = %20.15f, time_rec = %20.15f , Sp_glb = %20.15f\n", 
              t_dis, t_rec, sp_glb);
  }
  MPI_Finalize();
}