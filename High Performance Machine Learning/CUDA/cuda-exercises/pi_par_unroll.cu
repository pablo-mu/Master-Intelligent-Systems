#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#define MAX_THREADS 1024

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



template <unsigned int threads_per_block>
__device__ void warpReduce(volatile double* sdata, int tid) {
    if (threads_per_block >= 64) sdata[tid] += sdata[tid + 32];
    if (threads_per_block >= 32) sdata[tid] += sdata[tid + 16];
    if (threads_per_block >= 16) sdata[tid] += sdata[tid + 8];
    if (threads_per_block >= 8) sdata[tid] += sdata[tid + 4];
    if (threads_per_block >= 4) sdata[tid] += sdata[tid + 2];
    if (threads_per_block >= 2) sdata[tid] += sdata[tid + 1];
}


// Definition of CUDA kernels
template <unsigned int threads_per_block>
__global__ void pi_par_unroll(int num_steps, double step, double *pi) {
  int i;
  double x;
  extern __shared__ double sum[];
  unsigned int tid = threadIdx.x;
  unsigned int index = tid + blockIdx.x * (threads_per_block)*2;
  int gridsize = threads_per_block*2 *gridDim.x;

  // Initialize shared memory
  sum[tid] = 0.0;
  __syncthreads();

 
  // Local Accumulation
  /*
  for (i = index; i < num_steps; i += gridsize) {
    x = (i + 0.5) * step;
    sum[tid] += 4.0 / (1.0 + x * x);
  }
  */
  for (i = index; i < num_steps; i += gridsize) {
    x = (i + 0.5) * step;
    double sum1 = 4.0 / (1.0 + x * x);
    if (i + threads_per_block < num_steps) {
      x = (i + threads_per_block + 0.5) * step;
      double sum2 = 4.0 / (1.0 + x * x);
      sum[tid] += sum1 + sum2;
    } else {
        sum[tid] += sum1;
    }
  }

  __syncthreads();

  // Reduction in shared memory
  if (threads_per_block >= 512){if (tid < 256){sum[tid] += sum[tid+256];} __syncthreads();}
  if (threads_per_block >= 256){if (tid < 128){sum[tid] += sum[tid+128];} __syncthreads();}
  if (threads_per_block >= 128){if (tid < 64){sum[tid] += sum[tid+ 64];} __syncthreads();}
  if (tid < 32) warpReduce<threads_per_block>(sum, tid);
  if (tid == 0) atomicAdd(pi, step * sum[0]);
  //pi[blockIdx.x] = step*sum[0]; 
}


int main(int argc, char *argv[]) {
  double t_seq, t_par, sp, ep;

  // Adjust the number of active threads 
  // and the number of rectangules
  
  int num_steps = 100000;
  unsigned int threads_per_block = 128;
  int num_blocks = 1;
  if (argc == 2) {
    num_blocks = atoi(argv[1]);
  } else if (argc == 3) {
    num_blocks = atoi(argv[1]);
    num_steps = atoi(argv[2]);
  } else if (argc == 4) {
    threads_per_block = atoi(argv[1]);
    num_blocks = atoi(argv[2]);
    num_steps = atoi(argv[3]);
  }else if (argc > 4) {
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
  switch (threads_per_block){
    case 512:
      pi_par_unroll<512><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 256:
      pi_par_unroll<256><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 128:
      pi_par_unroll<128><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 64:
      pi_par_unroll< 64><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 32:
      pi_par_unroll<32><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 16:
      pi_par_unroll<16><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 8:
      pi_par_unroll<8><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 4:
      pi_par_unroll<4><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 2:
      pi_par_unroll<2><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
    case 1:
      pi_par_unroll<1><<< num_blocks, threads_per_block, threads_per_block*sizeof(double) >>>(num_steps, step, d_pi); break;
  }

  cudaEventRecord(stop_par, 0);  
  cudaEventSynchronize(stop_par);
  cudaEventElapsedTime(&time_par, start_par, stop_par);
  t_par = static_cast<double>(time_par) / 1000.0; 
  // Copy result back to host
  cudaMemcpy(pi_host, d_pi, sizeof(double), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_pi);

  sp = t_seq / t_par;
  ep = sp / threads_per_block;

  printf(" pi_par = %20.15f\n", *pi_host);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
