/* matrix summation using OpenMP

   usage with gcc (version 4.2 or higher required):
     gcc -O -fopenmp -o matrixSum-openmp matrixSum-openmp.c
     ./matrixSum-openmp size numWorkers

*/

#include <omp.h>
#include <stdio.h>

double start_time, end_time;

#define MAXGRIDSIZE 10000 /* maximum matrix size */
#define MAXWORKERS 8      /* maximum number of workers */
#define MAXITERATIONCOUNT 100000

int numWorkers, gridInteriorSize8h, gridInteriorSize4h, gridInteriorSize2h, gridInteriorSizeh, numIters; // "the grid size, **not including boundaries**"

void jacobi(double** grid1, double** grid2, int gridInteriorSize, int iterations){
    int x,y;
    for(int timesIterated = 0; timesIterated < iterations; timesIterated++){
        #pragma omp parallell for private(x) schedule(static)
        for(y=0; y < gridInteriorSize; y++)
        {
            grid2[y][x] = (grid1[y-1][x] + grid1[y+1][x] + grid1[y][x-1] + grid1[y][x+1]) / 4 ;
        }

        #pragma omp parallell for private(x) schedule(static)
        for(y=0; y < gridInteriorSize; y++)
        {
            grid1[y][x] = (grid2[y-1][x] + grid2[y+1][x] + grid2[y][x-1] + grid2[y][x+1]) / 4 ;
        }
    }
}

// "Restrict the result to coarse-grained (twice coarser) by a restriction operator" i.e. map fine-grained to less fine-grained
void restrictionCoarseify(double** fineGrid, double** coarseGrid, int interiorSizeCoarse){
    int yCoarse, xCoarse, yFine, xFine;

    #pragma omp parallel for private(xFine, yCoarse, xCoarse) // maybe schedule this?
    for(yCoarse = 1; yCoarse < interiorSizeCoarse; yCoarse++)
    {
        yFine = yCoarse << 1;
        for(xCoarse = 1; xCoarse < interiorSizeCoarse; xCoarse++)
        {
            xFine = xCoarse << 1;
            coarseGrid[yCoarse][xCoarse] = fineGrid[yFine][xFine]*0.5 + (fineGrid[yFine-1][xFine] + fineGrid[yFine][xFine-1] + fineGrid[yFine][xFine + 1] + fineGrid[yFine + 1][xFine]) * 0.125;
        }
    }

}

// "Interpolate the coarse grid back to the fine grid by using an interpolation operator"
void interpolateRefine(double** fineGrid, double** coarseGrid, int interiorSizeFine, int interiorSizeCoarse){
    int yCoarse, xCoarse, yFine, xFine;

    /* launch parallel threads*/
    #pragma omp parallel
    {
        /* Update the fine points that directly map to a coarse point in the grid */
        #pragma omp for private(xCoarse, yFine, xFine)
        for(yCoarse = 1; yCoarse < interiorSizeCoarse; yCoarse++)
        {
            yFine = yCoarse << 1;
            for(xCoarse = 1; xCoarse < interiorSizeCoarse; xCoarse++)
            {
                xFine = xCoarse << 1;
                fineGrid[yFine][xFine] = coarseGrid[yCoarse][xCoarse]; 
            }
        }
        /* Update the fine points that are in the same columns as a coarse point in the grid */
        #pragma omp for private(xCoarse, yFine, xFine)
        for(yCoarse = 1; yCoarse < interiorSizeFine; yCoarse += 2)
        {
            for(xCoarse = 2; xCoarse < interiorSizeFine; xCoarse += 2){
                fineGrid[yCoarse][xCoarse] = (fineGrid[yCoarse-1][xCoarse] + fineGrid[yCoarse+1][xCoarse]) * 0.5; 
            }
        }
        /* Update the rest of the fine points in the grid. */
        #pragma omp for private(xCoarse, yFine, xFine)
        for(yCoarse = 1; yCoarse < interiorSizeFine; yCoarse++)
        {
            for(xCoarse = 1; xCoarse < interiorSizeFine; xCoarse += 2){
                fineGrid[yCoarse][xCoarse] = (fineGrid[yCoarse][xCoarse-1] + fineGrid[yCoarse][xCoarse+1]) * 0.5;
            }
        }
    }
}

/* read command line, initialize, and create threads */
int main(int argc, char *argv[])
{
// ******** CONFIGURATION **********

    int x, y;

    /* read command line args if any */
    gridInteriorSize8h = (argc > 1) ? atoi(argv[1]) : MAXGRIDSIZE;
    numWorkers = (argc > 2) ? atoi(argv[2]) : MAXWORKERS;
    numIters = (argc > 3) ? atoi(argv[3]) : MAXITERATIONCOUNT;
    if (gridInteriorSize8h > MAXGRIDSIZE)
        gridInteriorSize8h = MAXGRIDSIZE;
    if (numWorkers > MAXWORKERS)
        numWorkers = MAXWORKERS;
    if (numIters > MAXITERATIONCOUNT)
        numIters = MAXITERATIONCOUNT;

    omp_set_num_threads(numWorkers);

    // Creating the grids (nested arrays). "Assume that all grids are square"
    int gridSideSize8h = gridInteriorSize8h + 2;

    // Interiors of the grids: "Assume that all grids are square. For the multigrid programs, the value of gridSize is the size of the coarsest (smallest) grid. The size of the next larger grid should then be 2*gridSize + 1, the next larger 2*(2*gridSize + 1) + 1, and so on."
    gridInteriorSize4h = (gridInteriorSize8h * 2) + 1;
    gridInteriorSize2h = (gridInteriorSize4h * 2) + 1;
    gridInteriorSizeh = (gridInteriorSize2h * 2) + 1;

    // As to be able to create squares of each grid
    int gridSideSize8h, gridSideSize4h, gridSideSize2h, gridSideSizeh; // since "gridSize" must be "not including boundaries"
    gridSideSize8h = gridInteriorSize8h + 2;
    gridSideSize4h = gridInteriorSize4h + 2;
    gridSideSize2h = gridInteriorSize2h + 2;
    gridSideSizeh = gridInteriorSizeh + 2;

    // Creating the 4 grids' rows: "For the multigrid programs, use a four-level V cycle"
    double** grid8h1 = malloc(gridSideSize8h * sizeof(double* ));
    double** grid8h2 = malloc(gridSideSize8h * sizeof(double* ));

    double** grid4h1 = malloc(gridSideSize4h * sizeof(double* ));
    double** grid4h2 = malloc(gridSideSize4h * sizeof(double* ));

    double** grid2h1 = malloc(gridSideSize2h * sizeof(double* ));
    double** grid2h2 = malloc(gridSideSize2h * sizeof(double* ));

    double** gridh1 = malloc(gridSideSizeh * sizeof(double* ));
    double** gridh2 = malloc(gridSideSizeh * sizeof(double* ));

    // Creating the 4 grids' columns:
    for (y = 0; y < gridSideSize8h; y++)
    {
        grid8h1[y] = malloc(gridSideSize8h * sizeof(double));
        grid8h2[y] = malloc(gridSideSize8h * sizeof(double));
    }
    for (y = 0; y < gridSideSize4h; y++)
    {
        grid4h1[y] = malloc(gridSideSize4h * sizeof(double));
        grid4h2[y] = malloc(gridSideSize4h * sizeof(double));
    }
    for (y = 0; y < gridSideSize2h; y++)
    {
        grid2h1[y] = malloc(gridSideSize2h * sizeof(double));
        grid2h2[y] = malloc(gridSideSize2h * sizeof(double));
    }
    for (y = 0; y < gridSideSizeh; y++)
    {
        gridh1[y] = malloc(gridSideSizeh * sizeof(double));
        gridh2[y] = malloc(gridSideSizeh * sizeof(double));
    }

    /* Fill the matrixes with their content */
    for (x = 0; x < gridSideSize8h; x++)
    {
        for (y = 0; y < gridSideSize8h; y++)
        {
            if (x == 0 || y == 0 || x == gridSideSize8h - 1 || y == gridSideSize8h - 1)
            {
                grid8h1[x][y] = 1;
                grid8h2[x][y] = 1;
            }
            else
            {
                grid8h1[x][y] = 0;
                grid8h2[x][y] = 0;
            }
        }
    }

    for (x = 0; x < gridSideSize4h; x++)
    {
        for (y = 0; y < gridSideSize4h; y++)
        {
            if (x == 0 || y == 0 || x == gridSideSize4h - 1 || y == gridSideSize4h - 1)
            {
                grid4h1[x][y] = 1;
                grid4h2[x][y] = 1;
            }

            else
            {
                grid4h1[x][y] = 0;
                grid4h2[x][y] = 0;
            }
        }
    }

    for (x = 0; x < gridSideSize2h; x++)
    {
        for (y = 0; y < gridSideSize2h; y++)
        {

            if (x == 0 || y == 0 || x == gridSideSize2h - 1 || y == gridSideSize2h - 1)
            {
                grid2h1[x][y] = 1;
                grid2h2[x][y] = 1;
            }

            else
            {
                grid2h1[x][y] = 0;
                grid2h2[x][y] = 0;
            }
        }
    }

    for (x = 0; x < gridSideSizeh; x++)
    {
        for (y = 0; y < gridSideSizeh; y++)
        {

            if (x == 0 || y == 0 || x == gridSideSizeh - 1 || y == gridSideSizeh - 1)
            {
                gridh1[x][y] = 1;
                gridh2[x][y] = 1;
            }
            else
            {
                gridh1[x][y] = 0;
                gridh2[x][y] = 0;
            }
        }
    }

// ******** MAIN PROGRAM EXECUTION **********
    start_time = omp_get_wtime();
  // CORSE-GRAINING: going down the "V"
        // "Start with a fine grid, and update points for a few iterations using any relaxation method (J, GS, SOR)"
    jacboi(gridh1, gridh2, gridInteriorSizeh, 4); // "Use exactly four iterations on each finer grid"
        // "Restrict the result to coarse-grained (twice coarser) by a restriction operator"
    restrictionCoarseify(gridh1, grid2h1, gridInteriorSize2h);

    jacobi(grid2h1, grid2h2, gridInteriorSize2h, 4);
    restrictionCoarseify(grid2h1, grid4h1, gridInteriorSize4h);

    jacobi(grid4h1, grid4h2, gridInteriorSize4h, 4);
    restrictionCoarseify(grid4h1, grid8h1, gridInteriorSize8h);

  // FINE-GRAINING: going up the "V"
    jacobi(grid8h1, grid8h2, gridInteriorSize8h, numIters); // "and use the command-line argument numIters (see below) for the number of iterations on the coarsest (smallest) grid."
            // "Interpolate the coarse grid back to the fine grid by using an interpolation operator"
    interpolation(grid4h1, grid8h1, gridInteriorSize4h, gridInteriorSize8h);

    jacobi(grid4h1, grid4h2, gridInteriorSize4h, 4);
    interpolation(grid2h1, grid4h1, gridInteriorSize2h, gridInteriorSize4h);

    jacobi(grid2h1, grid2h2, gridInteriorSize2h, 4);
    interpolation(gridh1, grid2h1, gridInteriorSizeh, gridInteriorSize2h);

    jacobi(gridh1, gridh2, gridInteriorSizeh, 4);

    end_time = omp_get_wtime();

// ******** MAXIMAL DIFFERENCE CALCULATION **********
    double curMaxDifference = 0.0;
    double curDifference;

    // TODO: Adapt it for this context

// ******** PRINTING THE RESULTS **********
    // TODO: adapt it for this context
    
    printf("Execution time: %g seconds\n", end_time - start_time);
    printf("Resulting maximal difference %d\n", curMaxDifference);

    FILE* outputFile = fopen("multigrid.out", "w");
    for (x = 0; x < gridInteriorSize8h; x++)
    {
        for (y = 0; y < gridInteriorSize8h; y++)
        {
            fprintf(outputFile, "%f ", grid1[x][y]);
        }
        fprintf(outputFile, "\n");
    }
    fclose(outputFile);

    return 0;
}
