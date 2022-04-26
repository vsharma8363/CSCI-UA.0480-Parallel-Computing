* Software Components
  * Blocks
  * Kernel
  * Threads
  * Grid
* Hardware Components
  * Warp --> Collection of threads
  * SM --> Streaming Multiprocessor
  * SP

* Restrictions of hardware/software
  * All threads execute same kernel
  * Dimensions of kernel can't change
  * Block must execute entirely by the SM
* Compute Capability
  * Number in form x.y
  * Way to expose hardware resources
  * **cudaGetDeviceProperties()**

#### cudaGetDeviceProperties()
* Called by the host to get info about the device properties

![cudaGetDeviceProperties](cuda_properties.png)

* you can also use **cudaGetDeviceCount(int* count)** to get num devices

```c
int numDevices;
cudaGetDeviceCount(numDevices);
for (int deviceID = 0; deviceID < numDevices; deviceID++) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceID);
  printf("Device ID %d is called %s\n", deviceID, prop.name);
}
```

#### Synchronization Points
```c
__syncthreads()
```
* Only threads **in the same block** can sync
 * Why? *to prevent a deadlock*
* Thread that makes call will be held at location until every thread reaches that location
* Be weary of if-then-else
 * IF sync threads are different, you will get a deadlock
* Ability to execute same app code on hardware with different number of execution resources --> transparent scalability

#### Scheduling of Blocks


