// must compile with: mpicc  -std=c99 -Wall -o checkdiv 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>


int main(int argc, char *argv[]){
  unsigned int x, A, B;
  FILE * fp; //for creating the output file
  char filename[100]=""; // the file name
  //char * numbers; //the numbers in the range [2, N]

  //double start_p2, end_p1;
  double time_for_p1 = 0.0; 

  //start_p1 = clock();
  // Check that the input from the user is correct.
  if(argc != 4){

    printf("usage:  ./checkdiv A B x\n");
    printf("A: the lower bound of the range [A,B]\n");
    printf("B: the upper bound of the range [A,B]\n");
    printf("x: divisor\n");
    exit(1);
  }  

  A = (unsigned int)atoi(argv[1]); 
  B = (unsigned int)atoi(argv[2]); 
  x = (unsigned int)atoi(argv[3]);
 

  // The arguments to the main() function are available to all processes and no need to send them from process 0.
  // Other processes must, after receiving the variables, calculate their own range.

  /////////////////////////////////////////

  // The main computation part starts here
  /////////////////////////////////////////
  //start of part 1

  // MPI Communicator Initialization procedures
  int my_rank, comm_sz;
  MPI_Init(&argc, &argv);
  double start_p1 = MPI_Wtime();
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Split the overall range [A, B] into segments (try to load balance)
  int numbers_per_process = (int)(((B-A)/comm_sz) + 1);
  int local_range_lower = A + (numbers_per_process * my_rank);
  int local_range_upper = local_range_lower + numbers_per_process;
  // If this is the last process, have it do the remaining numbers
  if (my_rank == comm_sz - 1)
    local_range_upper = B;
  else // If this isn't the last process, then increment upper range by 1
    local_range_upper = local_range_upper - 1;
  // Where to store divisible variables
  int *divisible_nums = malloc(sizeof(int) * (int)(numbers_per_process/2));
    for(int i = 0; i < (int)(numbers_per_process/2); i = i + 1){divisible_nums[i] = -1;}
  int total_divisible = 0;

  // Find all numbers that are divisible by x in range [local_range_lower, local_range_upper]
  for(int i = local_range_lower; i <= local_range_upper; i = i + 1){
    if (i % x == 0) { // divisible
      divisible_nums[total_divisible] = i;
      total_divisible = total_divisible + 1;
    }
  }

  // Combine all the divisible_nums arrays of length total_divisible in process 0
  // TODO(viksharma): Add way to concatenate arrays together.
  int *final_divisible_list = NULL;
  if (my_rank == 0) {
    final_divisible_list = malloc(sizeof(int) * (numbers_per_process*comm_sz));
    for(int i = 0; i < (numbers_per_process*comm_sz); i++)
      final_divisible_list[i] = -1;
    MPI_Gather(divisible_nums, (int)(numbers_per_process/2), MPI_INT, final_divisible_list, (int)(numbers_per_process/2), MPI_INT, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Gather(divisible_nums, (int)(numbers_per_process/2), MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // Use reduction operation to get MAX of (end_p1 - start_p1) among processes 
  // and send it to process 0 as time_for_p1
  double end_p1 = MPI_Wtime();
  double local_runtime_p1 = end_p1-start_p1;
  MPI_Reduce(&local_runtime_p1, &time_for_p1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  //end of part 1
  /////////////////////////////////////////


  /////////////////////////////////////////
  //start of part 2
  // Writing the results in the file
  if (my_rank == 0) {
    double start_p2 = MPI_Wtime();

    //forming the filename
    strcpy(filename, argv[2]);
    strcat(filename, ".txt");
    if( !(fp = fopen(filename,"w+t")))
    {
      printf("Cannot create file %s\n", filename);
      exit(1);
    }
    else {
      for(int i = 0; i < (numbers_per_process*comm_sz); i++) {
        if (final_divisible_list[i] != -1)
          fprintf(fp, "%d\n", final_divisible_list[i]);
          // TODO(viksharma): Write to file
      }
      //TODO(viksharma): Write the numbers divisible by x in the file as indicated in the lab description.
    }

    fclose(fp);

    double end_p2 = MPI_Wtime();
    //end of part 2
    /////////////////////////////////////////

    printf("time of part1 = %lf s    part2 = %lf s\n", 
          (double)(time_for_p1),
          (double)(end_p2-start_p2) );
  }
  MPI_Finalize();
  return 0;
}

