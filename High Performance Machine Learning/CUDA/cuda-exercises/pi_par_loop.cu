/*
Complete the file “pi_par_loop.cu”, so that each thread accumulates all its calculations 
on a local variable and the result is obtained by accumulating all these values.
First compute the reduction in each warp by means of a single loop, and use atomic 
operations to accumulate the local result on final pi
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS 1024

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


// Definition of CUDA kernels
__global__ void pi_par_loop(int num_steps, double step, double *pi) {
  int i;
  double x;
  extern __shared__ double sum[];
  unsigned int tid = threadIdx.x;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  int size = blockDim.x * gridDim.x;

  // Initialize shared memory
  sum[tid] = 0.0;
  __syncthreads();

  // Local Accumulation
  for (i = index; i < num_steps; i += size) {
    x = (i + 0.5) * step;
    sum[tid] += 4.0 / (1.0 + x * x);
  }
  __syncthreads();

  // Reduction in shared memory
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sum[tid] += sum[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    atomicAdd(pi, step * sum[0]);
  }
}


int main(int argc, char *argv[]) {
  double t_seq, t_par, sp, ep;

  // Adjust the number of active threads 
  // and the number of rectangules
  
  int num_steps = 100000;
  int threads = 128;
  int num_blocks = 1;
  if (argc == 2) {
    threads = atoi(argv[1]);
  } else if (argc == 3) {
    threads = atoi(argv[1]);
    num_blocks = atoi(argv[2]);
  } else if (argc == 4) {
    threads = atoi(argv[1]);
    num_blocks = atoi(argv[2]);
    num_steps = atoi(argv[3]);
  }
  else if (argc > 4) {
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
  float time_seq;
  float time_par;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  step = 1.0 / (double) num_steps;  
  for (i=0; i<num_steps; i++){
     x = (i+0.5)*step;
     sum = sum + 4.0/(1.0+x*x);
  }
  pi = step * sum;
  cudaEventRecord(stop, 0);  
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_seq, start, stop);
  t_seq = static_cast<double>(time_seq) / 1000.0; 

  printf(" pi_seq = %20.15f\n", pi);
  printf(" time_seq = %20.15f\n", t_seq);

  //
  // Parallel implementation
  //
  
  // Call to the CUDA
  double *pi_host; // Host copy of the variable pi
  double *d_pi; // Device copy of the variable pi

  // Allocate memory for device copies of pi
  cudaMalloc((void **)&d_pi, sizeof(double));
  //Initialize pi in device to 0
  cudaMemset(d_pi, 0, sizeof(double));
  // Allocate memory for host copy of pi and setup input values
  pi_host = (double *)malloc(sizeof(double));
  // Copy pi to device
  cudaMemcpy(d_pi, pi_host, sizeof(double), cudaMemcpyHostToDevice);
  // Launch pi_par_loop() kernel on GPU
  cudaEvent_t start_par, stop_par;
  cudaEventCreate(&start_par);
  cudaEventCreate(&stop_par);
  cudaEventRecord(start_par,0);
  pi_par_loop<<<num_blocks, threads, threads*sizeof(double)>>>(num_steps, step, d_pi);

  cudaEventRecord(stop_par, 0);  
  cudaEventSynchronize(stop_par);
  cudaEventElapsedTime(&time_par, start_par, stop_par);
  t_par = static_cast<double>(time_par) / 1000.0; 
  // Copy result back to host
  cudaMemcpy(pi_host, d_pi, sizeof(double), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_pi);

  sp = t_seq / t_par;
  ep = sp / threads;

  printf(" pi_par = %20.15f\n", *pi_host);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
