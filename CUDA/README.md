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
  
  int num_blocks = 1;
  Hello <<<num_blocks, thread_count>>> // Execute thread_count threads on GPU
  cudaDeviceSynchronize(); // Wait for GPU to be done
  
  return 0;
}
```
* This will **not** be executed in-order
* Kernel --> Code that will be offloaded to the GPU

### Breakdown
#### CUDA Kernel is executed by an array of threads
* All threads run same code --> SINGLE PROGRAM in SPMD
* Each thread --> ID
* ID = blockIdx.x * blockDim.x + threadIdx.x;
  * blockIdx.x (y or z)
  * blockDim.x (y or z)
  * threadIdx.x (y or z)
* Thread Blocks
  * Divide thread array into blocks
  * Threads within block cooperate with
    * Shared memory
    * Atomic ops
    * Barrier sync
  * Threads in different blocks **cannot** cooperate

### Levels of CUDA
1. Kernel
  * Launched by host
  * like C function
  * Executed on device
2. Grid
  * 1D, 2D, 3D
  * gridDim.x, gridDim.y, gridDim.z --> size of grid
3. Block
  * 1D, 2D, 3D
4. Thread

## IDs
* Thread uses ID to decide data to work on
  * Block ID: 1D, 2D, or 3D
  * Thread ID: 1D, 2D, or 3D

## Example 2 - Vector Addition
```c
#include <cuda.h>
void vecAdd(float* A, float* B, float* C) {
 // 1. Allocate device memory for A,B,C
 float* A_d, *B_d, *C_d;
  // Copy A and B to device memory
 cudaMalloc((void**)&A_d, size);
 cudaMemcpy(A_d, A, size, cudaMemcpyHosttoDevice);
 cudaMalloc((void**) &B_d, size);
 cudaMemcpy(B_d, B, size, cudaMemcopyHosttoDevice);
 cudaMalloc((void**)&C_d, size);
 
 // 2. Kernel launch code - Have device perform addition
 
 // 3. Copy C from device memory
 cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
 cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); 
}
```

### CUDA Memory Model
* Global Memory
  * Main means of communicating R/W Data between host and device
  * Contents visible to all threads
* Shared memory
  * Per SM
  * Shared by all threads in a block

#### CUDA Memory Allocation
* cudaMalloc()
  * Allocates object in *global memory*
  * Requires addy of pointer, size of object
* cudaFree()
  * Frees object from device Global memory

```c
WIDTH = 64;
float* Md;
int size = WIDTH * WIDTH * sizeof(float);
cudaMalloc((void**)&Md, size);
cudaFree(Md);
```

# Kernels
* ____global____ defines a kernel function - must **return void**
* ____device____ and *__host__* can be used together

## Hello World of Parallel Programming: Matrix Multiplication
* Data Parallelism --> Many operations on data structures at the same time
