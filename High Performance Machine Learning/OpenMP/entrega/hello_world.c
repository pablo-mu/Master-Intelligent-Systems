#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  
  // Adjust the number of active thrads
  if (argc == 2) {
    int param1 = atoi(argv[1]);
    omp_set_num_threads(param1);
  } else if (argc > 2) {
    printf("Wrong number of parameters\n");
    printf("./a.out [ num_threads ]\n");
    exit(-1);
  }

  // Parallel region
#pragma omp parallel default(none)
  {
    // Get the thread id
    int rank = omp_get_thread_num();
    // Get the number of threads
    int size = omp_get_num_threads();

    printf("Hello from thread %d of %d\n", rank, size);
  }
}

