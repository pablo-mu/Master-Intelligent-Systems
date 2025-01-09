/*
Complete the file “mat_vec.c”, so that all threads collaborate to compute a matrix-vector product. This file incorporates initial code for two solutions:
a) Add the #pragma in the external loop.
b) Add the #pragma in the internal loop
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  double t1, t2, t_seq, t_par, sp, ep;

  // Adjust the number of active threads 
  // and the number of rectangules
  int M = 100, N = 100;
  if (argc == 2) {
    int param1 = atoi(argv[1]);
    omp_set_num_threads(param1);
  } else if (argc == 3) {
    int param1 = atoi(argv[1]);
    omp_set_num_threads(param1);
    M = atoi(argv[2]);
  } else if (argc == 4) {
    int param1 = atoi(argv[1]);
    omp_set_num_threads(param1);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
  } else if (argc > 4) {
    printf("Wrong number of parameters\n");
    printf("./a.out [ num_threads [ M [ N ] ] ]\n");
    exit(-1);
  }

/*************************************/
/****** Computation of mat_mult ******/
/*************************************/

  int i, j;
	double A[M][N], x[N], y[M], sum, sum_seq, sum_par;

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

  //
  // Sequential implementation
  //
  t1 = omp_get_wtime();
	for (i=0; i<M; i++) {
		y[i] = 0;
	  for (j=0; j<N; j++) {
			y[i] += A[i][j] * x[j];
		}
	}
  t2 = omp_get_wtime();
  t_seq = (t2-t1);

	sum_seq = 0.0;
	for (i=0; i<M; i++) {
		sum_seq += y[i];
	}

  printf(" sum_seq = %20.15f\n", sum_seq);
  printf(" time_seq = %20.15f\n", t_seq);

  //
  // Parallel implementation
  //

  // Defining the number of active threads
  int size = 0;
#pragma omp parallel default(none) shared(size)
  {
    size = omp_get_num_threads();
  }
  
  // version a)

  t1 = omp_get_wtime();
	// Including the parallel for in the external loop
  #pragma omp parallel for default(none) shared(A, x, y, M, N) private(i, j)
  for (i=0; i<M; i++) {
    y[i] = 0;
    for (j=0; j<N; j++) {
      y[i] += A[i][j] * x[j];
    }
  }

  t2 = omp_get_wtime();
  t_par = (t2-t1);
  sp = t_seq / t_par;
  ep = sp/size;

	sum_par = 0.0;
	for (i=0; i<M; i++) {
		sum_par += y[i];
	}
  printf("External Loop");
  printf(" sum_par = %20.15f , diff = %20.15e\n", sum_par, sum_seq-sum_par);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);

  t1 = omp_get_wtime();
	// Including the parallel for in the internal loop
	for(i=0; i<M; i++) {
    double res = 0;
    #pragma omp parallel for default(none) shared(A, x, y, M, N,i) private(j) reduction(+:res)
    for (j=0; j<N; j++) {
      res += A[i][j] * x[j];
    }
    y[i] = res;
  }
  t2 = omp_get_wtime();
  t_par = (t2-t1);
  sp = t_seq / t_par;
  ep = sp/size;

	sum_par = 0.0;
	for (i=0; i<M; i++) {
		sum_par += y[i];
	}
  printf("Internal Loop");
  printf(" sum_par = %20.15f , diff = %20.15e\n", sum_par, sum_seq-sum_par);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
