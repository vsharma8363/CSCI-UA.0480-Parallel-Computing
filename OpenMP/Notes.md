# OpenMP Programming in C

## Basic Example
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
  // DOES NOT CREATE THREADS, sets the number of planned threads
  omp_set_num_threads(16);
  
  // Do this in parallel
  // Pragma --> "Create this"
  #pragma omp parallel
  {
    printf("Hello World!\n");
  }
  
  return 0;
}
```
* Only the code **in the brackets** will be in parallel, outside, it will collapse all threads
* It **adds 15 threads**, it doesn't create 16 - The master thread remains
* If you do not specify how many theads, it will set **num threads = num cores**

## Basics
* **API in C** for shared-memory systems
* Designed for **threading**
* OpenMP **does not run** on distributed memory systems
* Meant to parallelize **for-loops** originally
```c
void main() {
  double arr[1000];
  
  #pragma omp parallel for
  for(int i=0;i<1000;i++){
    do_something(arr[i]) 
  }
}
```
* Can parallelize many serial programs w/ FEW ANNOTATIONS
* OpenMP is a small API - Hides **cumbersome threading calls with simpler directives**

## Pragmas (#pragma)
* Special preprocessor instructions
* Specific by C standard
* Compilers that don't support pragma, ignore them
### Clauses
* Text that modifies a directive
* Example: num_threads
```c
#pragma omp parallel num_threads(10)
```
* num_threads is the clause

## Critical Sections - Mutex
```c
#pragma omp critical
global_result += my_result
```
* ONLY ONE THREAD can execute the structured block at a time

## Example - Getting thread rank
```c
void main() {
  int thread_count = 10;
  #pragma omp parallel num_threads(thread_count)
  Hello();
  
  return 0;
} 
void Hello(void) {
  int my_rank = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
}
```

## Running OpenMP Code
```shell
gcc -g -Wall -fopenmp -o omp_hello omp_hello.c
./omp_hello
```
