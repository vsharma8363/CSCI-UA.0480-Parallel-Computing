#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE 11.79


/*** TODO: insert the declaration of the kernel function below this line ***/
int numBlocks = 4;
int numThreads = 500;

__global__ void vecGPU(float *ad, float *bd, float *cd, int size, int elementsPerThread, int totalNumThreads) {
	// Calculate thread ID
	int threadID = threadIdx.x + blockDim.x * blockIdx.x;
	// Calculate upper and lower bound of calculation for this thread
	int lower = elementsPerThread * threadID;
	int upper = lower + elementsPerThread;
	if (threadID == (totalNumThreads) - 1)
		upper = size;
	// Run calculation procedure for each thread
	for(int i = lower; i < upper; i++) {
		if (i < size)
			cd[i] += ad[i] * bd[i];
	}
}
/**** end of the kernel declaration ***/


int main(int argc, char *argv[]){

	int n = 0; //number of elements in the arrays
	int i;  //loop index
	float *a, *b, *c; // The arrays that will be processed in the host.
	float *temp;  //array in host used in the sequential code.
	float *ad, *bd, *cd; //The arrays that will be processed in the device.
	clock_t start, end; // to meaure the time taken by a specific part of code
	
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
		}
		
	n = atoi(argv[1]);
	printf("Each vector will have %d elements\n", n);
	
	
	//Allocating the arrays in the host
	
	if( !(a = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}
	
	if( !(b = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array b\n");
	   exit(1);
	}
	
	if( !(c = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array c\n");
	   exit(1);
	}
	
	if( !(temp = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array temp\n");
	   exit(1);
	}
	
	//Fill out the arrays with random numbers between 0 and RANGE;
	srand((unsigned int)time(NULL));
	for (i = 0; i < n;  i++){
        a[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		b[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		c[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		temp[i] = c[i]; //temp is just another copy of C
	}
	
    //The sequential part
	start = clock();
	for(i = 0; i < n; i++) {
		temp[i] += a[i] * b[i];
	}
	end = clock();
	printf("Total time taken by the sequential part = %lf\n", (double)(end-start)/(double)CLOCKS_PER_SEC);

    /******************  The start GPU part: Do not modify anything in main() above this line  ************/
	//The GPU part
	start = clock();
	
	//	1. allocate ad, bd, and cd in the device
	int size = n*sizeof(float);
	cudaMalloc((void**)&ad, size);
	cudaMalloc((void**)&bd, size);
	cudaMalloc((void**)&cd, size);
	// 2. send a, b, and c to the device
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);
	// 4. Call the kernel (the kernel itself will be written at the comment at the end of this file), 
	//	   you need to write the number of threads, blocks, etc and their geometry.
	int elementsPerThread = (n / (numBlocks * numThreads));

	vecGPU<<<numBlocks, numThreads>>>(ad, bd, cd, n, elementsPerThread, numBlocks*numThreads);
	
	// 5. Bring the cd array back from the device and store it in c array (declared earlier in main)
	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
	// 6. free ad, bd, and cd
	cudaFree(ad); 
	cudaFree(bd);
	cudaFree(cd); 
	
	end = clock();
	printf("Total time taken by the GPU part = %lf\n", (double)(end-start)/ CLOCKS_PER_SEC);
	/******************  The end of the GPU part: Do not modify anything in main() below this line  ************/
	
	//checking the correctness of the GPU part
	for(i = 0; i < n; i++)
		if( fabsf(temp[i] - c[i]) >= 0.009) //compare up to the second degit in floating point
			printf("Element %d in the result array does not match the sequential version\n", i);
		
	// Free the arrays in the host
	free(a); free(b); free(c); free(temp);

	return 0;
}

/**** TODO: Write the kernel itself below this line *****/
