// Sequential implementation of histogram creation
//
// Usage: ./histogram b t filename
//  b - number of bins
//  t - number of threads
//  filename - name of file containing floating points

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

int what_bin(float range_of_bins, float number){
  // If the number is 100, return the last bin
  if(number == 100.0)
    return what_bin(range_of_bins, 99.0);
  return (int)(number/range_of_bins);
}

int main(int argc, char *argv[]){
    // Read input parameters
    unsigned int num_bins;
    num_bins = (unsigned int)atoi(argv[1]); 
    unsigned int num_threads;
    num_threads = (unsigned int)atoi(argv[2]); 
    char filename[100]="";
    strcpy(filename, argv[3]); 
    // Calculate range of bins
    float range_of_bins = (100.0/num_bins);

    // Reading all numbers from input file
    FILE * fp;
    int num_ints;
    float *numbers;
    if( !(fp = fopen(filename, "r")))
    {
      printf("Cannot read file %s\n", filename);
      exit(1);
    }
    else {
        fscanf(fp, "%d",&num_ints);
        numbers = malloc(sizeof(float) * num_ints);
        for (int i=0; i<num_ints; i++) {
          fscanf(fp, " %f", &numbers[i]);
        }
        fclose(fp);
    }

    omp_set_num_threads(num_threads);

    int (*bin)[num_bins] = calloc(num_bins, sizeof(int[num_threads][num_bins]));
    
    // Start time
    double start = omp_get_wtime();
    // Get ranges of integers to look at given thread count
    #pragma omp parallel
    {
      int my_rank = omp_get_thread_num();
      int numbers_per_thread = (int)(num_ints/num_threads);
      int lower_range = numbers_per_thread*my_rank;
      int upper_range = numbers_per_thread*(my_rank+1);
      if (my_rank == num_threads-1)
        upper_range = num_ints;
      // printf("Thread %d> Range is [%d, %d]\n", my_rank, lower_range, upper_range);
      for (int i=lower_range; i<upper_range; i++){
        int bin_number = what_bin(range_of_bins, numbers[i]);
        bin[my_rank][bin_number]++;
      }
    }
    double end = omp_get_wtime();
    // End time

    // Print result
    for(int bin_num=0; bin_num<num_bins; bin_num++) {
      int bin_val=0;
      for(int thread_num=0; thread_num<num_threads; thread_num++) {
        bin_val = bin_val + bin[thread_num][bin_num];
      }
      printf("bin[%d]=%d\n", bin_num, bin_val);
    }

    printf("Parallel part time = %1f s\n", (double)(end-start));
}