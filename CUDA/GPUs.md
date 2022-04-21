# GPUs - The Basics
* 3 Goals of Current Architectures
  * *Maintain execution speed* of old sequential programs (CPU)
  * *Increase throughput* of parallel programs (GPU)
  * *Reduce execution time* of non-GPU-friendly programs (Multicore) 
* **Throughput** --> How many threads finish work PER unit time
* Winning Apps --> CPUs + GPUs
  * CPUs do sequential parts
    * Low-level or no-data parallelism
    * 10x faster than GPUs for sequential
  * GPUs do parallel parts
    * 10x faster than CPUs for parallel
## GPU Structure
* GPU has it's own memory
  * Memory of GPU --> Optimized for bandwidth
  * SLOW from memory address to getting data
  * Why?
    * Trade-off between speed and bandwidth
    * Feed execution units with lots of data
* GPU compared to other chips
  * Good for subset of applications
  * Problem? Power consumption, liquid cooling needed
* Regularity + Massive Parallelism = GPU-Friendly
  * **Best performance:**
    * Computation intensive
    * Many independent computations
    * Many similar computations
    * Problem size is big enough
* Example: Matrix Multiplication

## PCIe
* 32 GB transfers per second per lane
* NVLINK --> Faster than PCIe
* Connects **CPU and GPU**

## GPU Programming Models
* Application --> Kernels --> Threads --> Blocks --> Grid
 * App --> Runs multiple kernels
 * Grid --> Executes a kernel 
 * Block --> 2D grid
 * Kernel --> Function executed on data



