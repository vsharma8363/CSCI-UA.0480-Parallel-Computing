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

![cudaGetDeviceProperties](cuda_structure.png)

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


