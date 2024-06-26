Parallel Jacobi solver for laplace equation in 2D

## Src
[paralleljacobi.c](paralleljacobi.c)

## Compilation
```bash
gcc -O2 -o paralleljacobi paralleljacobi.c -lm -fopenmp
```

## Run

Parameters:
- grid_size: size of the grid
- num_iterations: number of iterations
- num_threads: number of threads to use

Example:
```bash
./paralleljacobi 200 1000 4 
```