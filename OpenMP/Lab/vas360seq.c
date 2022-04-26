// Sequential implementation of histogram creation
//
// Usage: ./histogram b filename
//  b - number of bins
//  filename - name of file containing floating points

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
    char filename[100]="";
    strcpy(filename, argv[2]); 
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

    clock_t start = clock();
    
    // Find and store bin ranges
    int *bin = calloc(num_bins, sizeof(int));
    for (int i=0; i<num_ints; i++){
      int bin_number = what_bin(range_of_bins, numbers[i]);
      bin[bin_number]++;
    }

    clock_t end = clock();

    // Print result
    for(int i=0; i<num_bins; i++){
      printf("bin[%d] = %d\n", i, bin[i]);
    }

    printf("Total time for sequential = %f s\n", (double)(end-start)/CLOCKS_PER_SEC);
}