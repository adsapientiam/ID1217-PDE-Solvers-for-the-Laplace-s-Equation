// Sequental Jacobi method for solving linear systems of equations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define MAXGRIDSIZE 10000        /* maximum matrix size */
#define MAXITERATIONCOUNT 1000000 /* maximum number of iterations */
#define MAXWORKERS 8             /* maximum number of workers */
#define CONVERGENCE_THRESHOLD 0.0001 /* maximum difference in grid */
#define USE_CONVERGENCE false /* whether to use the convergence threshold */

// Struct to hold the parsed arguments
struct args
{
    int gridInteriorSize;
    int numIters;
    int numWorkers;
};

// Function to parse the arguments
struct args parseArgs(int argc, char *argv[]) {
    struct args parsedArgs;
    parsedArgs.gridInteriorSize = (argc > 1) ? atoi(argv[1]) : MAXGRIDSIZE;
    parsedArgs.numIters = (argc > 2) ? atoi(argv[2]) : MAXITERATIONCOUNT;
    parsedArgs.numWorkers = (argc > 3) ? atoi(argv[3]) : MAXWORKERS;
    if (parsedArgs.gridInteriorSize > MAXGRIDSIZE)
        parsedArgs.gridInteriorSize = MAXGRIDSIZE;
    if (parsedArgs.numIters > MAXITERATIONCOUNT)
        parsedArgs.numIters = MAXITERATIONCOUNT;
    if (parsedArgs.numWorkers > MAXWORKERS)
        parsedArgs.numWorkers = MAXWORKERS;
    return parsedArgs;
}

// Function to initialize the grid
void initializeGrid(int gridInteriorSize, double (*grid)[gridInteriorSize + 2], double (*newGrid)[gridInteriorSize + 2]) {
    for (int i = 0; i < gridInteriorSize + 2; i++) {
        for (int j = 0; j < gridInteriorSize + 2; j++) {
            if (i == 0 || j == 0 || i == gridInteriorSize + 1 || j == gridInteriorSize + 1) {
                grid[i][j] = 1.0;
                newGrid[i][j] = 1.0;
            } else {
                grid[i][j] = 0.0;
                newGrid[i][j] = 0.0;
            }
        }
    }
}

// Function to perform the Jacobi iteration
double jacobiIteration(int gridInteriorSize, double (*grid)[gridInteriorSize + 2], double (*newGrid)[gridInteriorSize + 2], int numIters) {
    double diff; /* difference between old and new grid */
    double maxDiff; /* maximum difference in grid */

    /* iterate */
    for (int k = 0; k < numIters; k++) {
        maxDiff = 0.0;
        for (int i = 1; i <= gridInteriorSize; i++) {
            for (int j = 1; j <= gridInteriorSize; j++) {
                newGrid[i][j] = 0.25 * (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1]);
                diff = fabs(newGrid[i][j] - grid[i][j]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

        // Swap the grids
        double (*tempGrid)[MAXGRIDSIZE + 2] = grid;
        grid = newGrid;
        newGrid = tempGrid;

        // If the maximum difference is less than 0.0001, we can stop iterating
        if (maxDiff < CONVERGENCE_THRESHOLD && USE_CONVERGENCE) {
            break;
        }
    }

    return maxDiff;
}

// Function to calculate the maximum error
double calculateMaxError(int gridInteriorSize, double (*grid)[gridInteriorSize + 2]) {
    double maxError = 0.0;
    for (int i = 1; i < gridInteriorSize + 1; i++) {
        for (int j = 1; j < gridInteriorSize + 1; j++) {
            double error = fabs(grid[i][j] - 1.0);
            if (error > maxError) {
                maxError = error;
            }
        }
    }
    return maxError;
}

// Function to write the grid to a file
void writeGridToFile(int gridInteriorSize, double (*grid)[gridInteriorSize + 2]) {
    FILE *file = fopen("filedata.out", "w");
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }
    for (int i = 0; i < gridInteriorSize + 2; i++) {
        for (int j = 0; j < gridInteriorSize + 2; j++) {
            if (fprintf(file, "%f ", grid[i][j]) < 0) {
                printf("Error writing to file.\n");
                fclose(file);
                return;
            }
        }
        if (fprintf(file, "\n") < 0) {
            printf("Error writing to file.\n");
            fclose(file);
            return;
        }
    }
    if (fclose(file) != 0) {
        printf("Error closing file.\n");
        return;
    }
}

// Main function
int main(int argc, char *argv[]) {
    struct args parsedArgs = parseArgs(argc, argv);
    
    // Creating the grids, assume that all grids are square
    // The size of the grid is the interior size plus 2 for the boundaries
    double (*grid)[parsedArgs.gridInteriorSize + 2] = malloc(sizeof(double[parsedArgs.gridInteriorSize + 2][parsedArgs.gridInteriorSize + 2]));
    double (*newGrid)[parsedArgs.gridInteriorSize + 2] = malloc(sizeof(double[parsedArgs.gridInteriorSize + 2][parsedArgs.gridInteriorSize + 2]));

    if (grid == NULL || newGrid == NULL) {
        printf("Error: Failed to allocate memory for grid.\n");
        return 1;
    }

    /* initialize grid */
    // The grid is initialized to 0.0, except for the boundaries which are initialized to 1.0
    initializeGrid(parsedArgs.gridInteriorSize, grid, newGrid);

    /* start timer */
    struct timespec start, end;
    timespec_get(&start, TIME_UTC);

    /* iterate */
    double maxDiff = jacobiIteration(parsedArgs.gridInteriorSize, grid, newGrid, parsedArgs.numIters);

    /* end timer */
    timespec_get(&end, TIME_UTC);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1E9;

    // Calculate max error
    double maxError = calculateMaxError(parsedArgs.gridInteriorSize, grid);

    // Print args
    printf("Grid size: %d\n", parsedArgs.gridInteriorSize);
    printf("Number of workers: %d\n", parsedArgs.numWorkers);
    printf("Number of iterations: %d\n", parsedArgs.numIters);

    // Print execution time
    printf("Execution time: %f seconds\n", time_taken);

    // Print maximum error 
    printf("Maximum error: %f\n", maxError);

    // Write the grid to a file
    writeGridToFile(parsedArgs.gridInteriorSize, grid);

    // Clean up
    free(grid);
    free(newGrid);

    return 0;
}