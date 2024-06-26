// Multigrid method for solving linear systems of equations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#define MAXGRIDSIZE 10000        /* maximum matrix size */
#define MAXITERATIONCOUNT 1000000000 /* maximum number of iterations */
#define MAXWORKERS 8             /* maximum number of workers */
#define CONVERGENCE_THRESHOLD 0.0001 /* maximum difference in grid */
#define USE_CONVERGENCE false /* whether to use the convergence threshold */
#define V_CYCLE_LEVELS 4 /* number of levels in the V-cycle */
#define CACHE_LINE_SIZE 64 /* cache line size in bytes */
#define DOUBLE_SIZE 8 /* size of double in bytes */
#define PADDING (CACHE_LINE_SIZE / DOUBLE_SIZE) /* padding to avoid false sharing */

#define min(a, b) ((a) < (b) ? (a) : (b)) /* minimum of two numbers */

// Struct to hold the parsed arguments
typedef struct args {
    int gridInteriorSize;
    int numIters;
    int numWorkers;
} Args;

// Struct to avoid false sharing
typedef struct {
    double value;
    char padding[PADDING];
} PaddedDouble;

typedef struct grid {
    int size;
    PaddedDouble** data;
} Grid;

typedef struct multigrid {
    Grid** grids;
    Grid** newGrids;
    int numGrids;
} Multigrid;

// Function to parse the arguments
Args parseArgs(int argc, char *argv[]) {
    Args parsedArgs;
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

// Allocate grid of size gridSize
PaddedDouble** allocateGrid(int gridSize) {
    PaddedDouble* data = (PaddedDouble*)malloc(gridSize * gridSize * sizeof(PaddedDouble));
    PaddedDouble** grid = (PaddedDouble**)malloc(gridSize * sizeof(PaddedDouble*));
    for (int i = 0; i < gridSize; i++) {
        grid[i] = &(data[i * gridSize]);
    }
    return grid;
}

// Copy grid
Grid* copyGrid(Grid* grid) {
    Grid* newGrid = (Grid*)malloc(sizeof(Grid));
    newGrid->size = grid->size;
    newGrid->data = allocateGrid(grid->size);
    for (int i = 0; i < grid->size; i++) {
        for (int j = 0; j < grid->size; j++) {
            newGrid->data[i][j].value = grid->data[i][j].value;
        }
    }
    return newGrid;
}

// Free multigrid and grids
void freeMultigrid(Multigrid* multigrid) {
    for (int i = 0; i < multigrid->numGrids; i++) {
        free(multigrid->grids[i]->data[0]);
        free(multigrid->grids[i]->data);
        free(multigrid->grids[i]);
        free(multigrid->newGrids[i]->data[0]);
        free(multigrid->newGrids[i]->data);
        free(multigrid->newGrids[i]);
    }
    free(multigrid->grids);
    free(multigrid->newGrids);
}

// Initialize a grid of size gridSize, with all elements set to 0, except for the boundary elements, which are set to 1
void initializeGrid(int gridSize, PaddedDouble** grid) {
    
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            grid[i][j].value = 0;
        }
    }

    for (int i = 0; i < gridSize; i++) {
        grid[i][0].value = 1;
        grid[i][gridSize - 1].value = 1;
        grid[0][i].value = 1;
        grid[gridSize - 1][i].value = 1;
    }
}

// Smoothing function using Jacobi method
void jacobiSmoothing(Grid* grid, Grid* newGrid) {
    int gridSize = grid->size;

    #pragma omp parallel for
    for (int i = 1; i < gridSize - 1; i++) {
        for (int j = 1; j < gridSize - 1; j++) {
            newGrid->data[i][j].value = (grid->data[i - 1][j].value + grid->data[i + 1][j].value + grid->data[i][j - 1].value + grid->data[i][j + 1].value) / 4;
        }
    }
}

// Grid swap function (swap the data pointers of two grids)
void swapGrids(Grid* grid1, Grid* grid2) {
    PaddedDouble** temp = grid1->data;
    grid1->data = grid2->data;
    grid2->data = temp;
}

// Restriction function
// coarse[x,y] = fine[i,j]*0.5 + (fine[i-1,j]+fine[i,j-1]+fine[i+1,j]+fine[i,j+1])*0.125
void restriction(Grid* fineGrid, Grid* coarseGrid) {
    int coarseSize = coarseGrid->size;
    int fineSize = fineGrid->size;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < coarseSize - 1; i++) {
        for (int j = 1; j < coarseSize - 1; j++) {
            int x = min(i * 2, fineSize - 2);
            int y = min(j * 2, fineSize - 2);

            coarseGrid->data[i][j].value = fineGrid->data[x][y].value * 0.5 + (fineGrid->data[x - 1][y].value + fineGrid->data[x][y - 1].value + fineGrid->data[x + 1][y].value + fineGrid->data[x][y + 1].value) * 0.125;
        }
    }
}

// Prolongation function (bilinear interpolation)
void prolongation(Grid* coarseGrid, Grid* fineGrid) {
    int coarseSize = coarseGrid->size;
    int fineSize = fineGrid->size;
    
    // Copy coarse grid to fine grid
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < coarseSize - 1; i++) {
        for (int j = 1; j < coarseSize - 1; j++) {
            int x = i * 2;
            int y = j * 2;
            fineGrid->data[x][y].value = coarseGrid->data[i][j].value;
        }
    }

    // Interpolate
    #pragma omp parallel for collapse(2)
    for (int i = 2; i < fineSize - 2; i += 2) {
        for (int j = 2; j < fineSize - 2; j += 2) {
            fineGrid->data[i - 1][j].value = 0.5 * (fineGrid->data[i - 2][j].value + fineGrid->data[i][j].value);
            fineGrid->data[i][j - 1].value = 0.5 * (fineGrid->data[i][j - 2].value + fineGrid->data[i][j].value);
            fineGrid->data[i - 1][j - 1].value = 0.25 * (fineGrid->data[i - 2][j - 2].value + fineGrid->data[i][j - 2].value + fineGrid->data[i - 2][j].value + fineGrid->data[i][j].value);
        }
    }
}

// Iterative V-cycle function (4 levels)
void vCycle(Multigrid* multigrid, int numIters) {

    // For each level (going down to the coarsest grid)
    for (int i = V_CYCLE_LEVELS - 1; i > 0; i--) {
        
        // Pre-smoothing (4 iterations)
        for (int j = 0; j < 4; j++) {
            jacobiSmoothing(multigrid->grids[i], multigrid->newGrids[i]);
            swapGrids(multigrid->grids[i], multigrid->newGrids[i]);
        }

        // Residual calculation not needed for laplace equation

        // Restriction (relaxation)
        restriction(multigrid->grids[i], multigrid->grids[i - 1]);
    }

    // Coarsest grid
    // Solve with numIters iterations
    for (int i = 0; i < numIters; i++) {
        jacobiSmoothing(multigrid->grids[0], multigrid->newGrids[0]);
        swapGrids(multigrid->grids[0], multigrid->newGrids[0]);
    }

    // For each level (going up to the finest grid)
    for (int i = 0; i < V_CYCLE_LEVELS - 1; i++) {
        
        // Create copy of the fine grid
        Grid* interPolatedGrid = copyGrid(multigrid->grids[i+1]);

        // Prolongation
        prolongation(multigrid->grids[i], interPolatedGrid);

        // Correction
        for (int j = 0; j < multigrid->grids[i+1]->size; j++) {
            for (int k = 0; k < multigrid->grids[i+1]->size; k++) {
                multigrid->grids[i+1]->data[j][k].value += interPolatedGrid->data[j][k].value;
            }
        }

        // Free interpolated grid
        free(interPolatedGrid->data[0]);
        free(interPolatedGrid->data);
        free(interPolatedGrid);

        // Post-smoothing (4 iterations)
        for (int j = 0; j < 4; j++) {
            jacobiSmoothing(multigrid->grids[i+1], multigrid->newGrids[i+1]);
            swapGrids(multigrid->grids[i+1], multigrid->newGrids[i+1]);
        }
    }
}

// Function to calculate the maximum error
double calculateMaxError(Grid* grid1) {
    double maxError = 0;
    for (int i = 1; i < grid1->size - 1; i++) {
        for (int j = 1; j < grid1->size - 1; j++) {
            double error = fabs(grid1->data[i][j].value - 1);
            if (error > maxError) {
                maxError = error;
            }
        }
    }
    return maxError;
}

// Write the grid to a file
void writeGridToFile(Grid* grid) {
    FILE* file = fopen("filedata.out", "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        return;
    }
    for (int i = 0; i < grid->size; i++) {
        for (int j = 0; j < grid->size; j++) {
            fprintf(file, "%f ", grid->data[i][j].value);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Allocate multigrid and grids
void allocateMultigrid(Multigrid* multigrid, int gridSize) {
    multigrid->numGrids = V_CYCLE_LEVELS;
    multigrid->grids = (Grid**)malloc(V_CYCLE_LEVELS * sizeof(Grid*));
    multigrid->newGrids = (Grid**)malloc(V_CYCLE_LEVELS * sizeof(Grid*));

    for (int i = V_CYCLE_LEVELS - 1; i >= 0; i--) {
        int currentGridSize = (1 << i) * gridSize; // Left shift by i is equivalent to multiplying by 2^i
        
        Grid* grid = (Grid*)malloc(sizeof(Grid));
        grid->data = allocateGrid(currentGridSize);
        grid->size = currentGridSize;
        multigrid->grids[i] = grid;

        Grid* newGrid = (Grid*)malloc(sizeof(Grid));
        newGrid->data = allocateGrid(currentGridSize);
        newGrid->size = currentGridSize;
        multigrid->newGrids[i] = newGrid;
    }
}

// Main function for the multigrid method, using a V-cycle and Jacobi smoothing
int main(int argc, char *argv[]) {
    Args parsedArgs = parseArgs(argc, argv);
    
    // Allocate the grids. 
    // The first grid is the coarsest grid, and the last grid is the finest grid
    Multigrid multigrid;
    allocateMultigrid(&multigrid, parsedArgs.gridInteriorSize + 2);

    // Initialize all grids
    for (int i = 0; i < V_CYCLE_LEVELS; i++) {
        initializeGrid(multigrid.grids[i]->size, multigrid.grids[i]->data);
        initializeGrid(multigrid.newGrids[i]->size, multigrid.newGrids[i]->data);
    }
    
    // Set threads
    omp_set_num_threads(parsedArgs.numWorkers);

    /* start timer */
    struct timespec start, end;
    timespec_get(&start, TIME_UTC);

    // Perform the V-cycle
    vCycle(&multigrid, parsedArgs.numIters);

    /* end timer */
    timespec_get(&end, TIME_UTC);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1E9;

    // Calculate max error
    double maxError = calculateMaxError(multigrid.grids[0]);

    // Print args
    printf("Grid size: %d\n", parsedArgs.gridInteriorSize);
    printf("Number of workers: %d\n", parsedArgs.numWorkers);
    printf("Number of iterations: %d\n", parsedArgs.numIters);

    // Print execution time
    printf("Execution time: %f seconds\n", time_taken);

    // Print maximum error 
    printf("Maximum error: %f\n", maxError);

    // Write the grid to a file
    writeGridToFile(multigrid.grids[0]);

    // Clean up
    freeMultigrid(&multigrid);

    return 0;
}