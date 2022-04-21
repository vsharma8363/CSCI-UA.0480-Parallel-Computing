# GPU Programming with CUDA

## Parallel Computing on GPUs
* 13.8k GFLOPS
  * Laptops, clusters, etc.
* GPU parallelism *doubles* each year
* Programming model scales
  * Data prallelism
* Programmable in **C/C++**
* Multithreaded SPMD model
  * Single Program Multiple Data

## CUDA - Compute Unified Device Architecture
* Integrated host+device app C Program
  * Serial or *small* parallel parts - **C code**
  * Highly parallel parts - **SPMD kernel C code** 

## Example
```c
#include <stdio.h>
#include <cuda.h>

/* Device code: runs on GPU */
__global__ void Hello(void) {
  printf("Hello from thread %d\n", threadIdx.x);
}

/* Host code: runs on CPU */
int main(int argc, char* argv[]) {
  int thread_count; // Number of threads to run
  thread_count = strtol(argv[1], NULL, 10); // Get thread count from cmd line
  
  Hello <<<1, thread_count>>> // Execute thread_count threads on GPU
  cudaDeviceSynchronize(); // Wait for GPU to be done
  
  return 0;
}
```
* 
