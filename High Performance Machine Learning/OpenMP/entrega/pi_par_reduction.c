// Use the reduction clause to parallelize the computation of pi
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  double t1, t2, t_seq, t_par, sp, ep;

  // Adjust the number of active threads 
  // and the number of rectangules
  int num_steps = 100000;
  if (argc == 2) {
    int param1 = atoi(argv[1]);
    omp_set_num_threads(param1);
  } else if (argc == 3) {
    int param1 = atoi(argv[1]);
    omp_set_num_threads(param1);
    num_steps = atoi(argv[2]);
  } else if (argc > 3) {
    printf("Wrong number of parameters\n");
    printf("./a.out [ num_threads [ num_steps ] ]\n");
    exit(-1);
  }

/*************************************/
/******** Computation of pi **********/
/*************************************/

  int i;
  double step = 1.0 / (double) num_steps;  
  double pi = 0.0;

  //
  // Sequential implementation
  //
  double x, sum = 0.0;
  t1 = omp_get_wtime();
  step = 1.0 / (double) num_steps;  
  for (i=0; i<num_steps; i++){
     x = (i+0.5)*step;
     sum = sum + 4.0/(1.0+x*x);
  }
  pi = step * sum;
  t2 = omp_get_wtime();
  t_seq = (t2-t1);

  printf(" pi_seq = %20.15f\n", pi);
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
  
  pi = 0.0;
  t1 = omp_get_wtime();
  // version a)
#pragma omp parallel for default(none) shared(num_steps, step, size) private(x) reduction(+:pi)
  for (int i=0 ; i<num_steps; i++){
    x = (i+0.5)*step;
    pi += 4.0/(1.0+x*x) * step;
  }

  t2 = omp_get_wtime();
  t_par = (t2-t1);
  sp = t_seq / t_par;
  ep = sp/size;

  printf("version a)\n");
  printf(" pi_par = %20.15f\n", pi);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);

  pi = 0.0;
  // version b)
  #pragma omp parallel default(none) shared(num_steps, step, size, pi) private(x, sum)
  { 
    sum = 0.0; 
    #pragma omp for 
    for (int i=0; i<num_steps; i++){
      x = (i+0.5)*step; 
      sum += 4.0/(1.0+x*x)*step;
    }
  #pragma omp critical
  pi += sum;
  }

  t2 = omp_get_wtime();
  t_par = (t2-t1);
  sp = t_seq / t_par;
  ep = sp/size;

  printf("version b)\n");
  printf(" pi_par = %20.15f\n", pi);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
