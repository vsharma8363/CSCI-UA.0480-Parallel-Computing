# Cuda III
* Threads/block need to be as close as possible to multiple of 32

##### Quick Exercises
* If CUDA device's SM can take 1536 threads and 4 blocks, what config would result in most number of threads in the SM?
  * 128 threads/blk
  * 256 threads/blk
  * **512 threads/blk**
  * 1024 threads/blk
* 3*512 = 1536 threads, 3 < 4

### Computation vs. Memory Access
* CGMA (Compute to global memory access) ratio
  * For every floating point calculating you read from memory, how long before you get another number (you are done computing with it)

### GPU Breakdown
* Registers
  * Fastest
  * Does not consume off-chip bandiwdth, lifetime of thread
* Shared Memory
  * Extremely Fast
  * Highly parallel
  * Restricted to block
* Global Memory
  * High Access Latency: 400-800 cycles
  * Implmented in DRAM
  * Throughput: up to 936.2GB/s
* Constant Memory
  * Short latency, high bandwidth

* Each access to registers involves fewer machine-level instructions
* Aggregate register files --> 2 orders of magnitude of global

Announcements:
- Only Cuda III is covered on final
- Previous exams will be posted today without solutions
- Exam: May 12th (time of lecture) - 2pm
 - 24 hours to solve it
- Lab 3:
 -  In real cases: You need to include memcopy, malloc
 -  Just put start/stop around the kernel
 -  --> 1) Make sure the kernel runs (make a printf or something)
 -  --> 2) If the kernel launches, then include the cudamemcpy back or you can put after the kernel,
 -  Start --> Kernel --> cudaDeviceSynchronize (blocking the host until the device finishes) --> Stop
 -  It doesn't matter if you have speedup or slowdown
