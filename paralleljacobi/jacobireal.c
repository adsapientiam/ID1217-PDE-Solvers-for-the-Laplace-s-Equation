/* matrix summation using OpenMP

   usage with gcc (version 4.2 or higher required):
     gcc -O -fopenmp -o matrixSum-openmp matrixSum-openmp.c
     ./matrixSum-openmp size numWorkers

*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double start_time, end_time;

#define MAXGRIDSIZE 10000 /* maximum matrix size */
#define MAXWORKERS 8      /* maximum number of workers */
#define MAXITERATIONCOUNT 100000

int numWorkers, gridInteriorSize, numIters; // "the grid size, **not including boundaries**"

/* read command line, initialize, and create threads */
int main(int argc, char* argv[])
{
// ******** CONFIGURATION **********

    int x, y;
    double** temp;

    /* read command line args if any */
    gridInteriorSize = (argc > 1) ? atoi(argv[1]) : MAXGRIDSIZE;
    numWorkers = (argc > 2) ? atoi(argv[2]) : MAXWORKERS;
    numIters = (argc > 3) ? atoi(argv[3]) : MAXITERATIONCOUNT;
    if (gridInteriorSize > MAXGRIDSIZE) gridInteriorSize = MAXGRIDSIZE;
    if (numWorkers > MAXWORKERS) numWorkers = MAXWORKERS;
    if (numIters > MAXITERATIONCOUNT) numIters = MAXITERATIONCOUNT;

    omp_set_num_threads(numWorkers);

    // Creating the grids (nested arrays). "Assume that all grids are square"
    int gridSideSize = gridInteriorSize+2; // since "gridSize" must be "not including boundaries"
    // the rows along which columns will be added
    double** grid = malloc(gridSideSize*sizeof(double*)); // as to be able to create variable size multidimensional arrays
    double** new = malloc(gridSideSize*sizeof(double*));
    for(x=0; x < gridSideSize; x++){ // the columns
        grid[x] = malloc(gridSideSize * sizeof(double)); // most likely wrong way of doing it. Also, maye preferable to do "Using a single pointer and a 1D array with pointer arithmetic as per "https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/
        new[x] = malloc(gridSideSize * sizeof(double));
    }

    /* initialize the matrix' contents */
    for (x = 0; x < gridSideSize; x++)
    {
        //  printf("[ ");
        for (y = 0; y < gridSideSize; y++)
        {
            if(x == 0 || y == 0 || x == gridInteriorSize+1 || y == gridSideSize+1){
                grid[y][x] = 1.0;
                new[y][x] = 1.0;
            }
            else
            {
                grid[y][x] = 0.0;
                new[y][x] = 0.0;
            }
            
            //	  printf(" %d", matrix[i][j]);
        }
        //	  printf(" ]\n");
    }

// ******** JACOBI PROGRAM EXECUTION **********
    start_time = omp_get_wtime();
    
    for(int timesIterated=0; timesIterated < numIters; timesIterated++){
        // stripify? => private(y, rowMultiplier) and thus nested loop
            // use the `omp_get_num_threads();` to create the intervals 
            // if so, the worksharing construct should instead be (gridSideSize / numWorkers) such that nested for loop in the intervals
                // maye preferable to do "Using a single pointer and a 1D array with pointer arithmetic:" https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/ then?
                // Is it supposed to be determenistic? How then will the implementation of the barrier not make so it just is synchronous row-for-row?
        #pragma omp parallel for private(x) schedule(static) // Should satisfy "The largest strip should have at most one more row than the smallest strip".
        for (y = 1; y < gridInteriorSize; y++)            // If not, remove the "schedule(static)" and make the team-making for loop into for(int segmentStart = 0; segmentStart < omp_get_thread_num(); segmentStart++); and make the contents into a nested for loop in which the outer for loop iterates `for(int y = gridSize/omp_get_num_threads(); y < gridSize/omp_get_num_threads() * (omp_get_thread_num() + 1); y++)` and then what is written below as its inner loop. Have some condition emdedded in the y-loop's condition such that it will only have at most one
        {
            // "Compute new values for all interior points"
            for (x = 1; x < gridInteriorSize; x++) 
            {
                new[y][x] = (grid[y-1][x] + grid[y+1][x] + grid[y][x-1] + grid[y][x+1]) / 4 ;
            } 

        }
        // implicit barrier

        #pragma omp parallel for private(x) schedule(static) 
        for (y = 1; y < gridInteriorSize; y++)
        {
            // "Compute new values again for interior points"
            for (x = 1; x < gridInteriorSize; x++) // "+1" because `x` begins at `x=1` instead of `x=0`
            {
                grid[y][x] = (new[y-1][x] + new[y+1][x] + new[y][x-1] + new[y][x+1]) / 4 ;
            } 
        }
        // Implicit barrier
    }
    
    end_time = omp_get_wtime();

// ******** MAXIMAL DIFFERENCE CALCULATION **********
  // "Compute the maximum difference"
  double curMaxDifference = 0.0;
  double curDifference;
  for (x = 1; x < gridInteriorSize; x++) {
    for (y = 1; y < gridInteriorSize; y++) {
      curDifference = (grid[y][x] - new[y][x]);
      if(curDifference < 0) curDifference = (-1)*curDifference;
      if(curDifference > curMaxDifference) curMaxDifference = curDifference;
    }
  }

// ******** PRINTING THE RESULTS **********
    printf("Execution time: %g seconds\n", end_time - start_time);
    printf("Resulting maximal difference %d\n", curMaxDifference);

    FILE* outputFile = fopen("jacobi.out", "w");
    for (x = 0; x < gridSideSize; x++)
    {
        for (y = 0; y < gridSideSize; y++)
        {
            fprintf(outputFile, "%f ", grid[y][x]);
        }
        fprintf(outputFile, "\n");
    }
    fclose(outputFile);

    return 0;
}
