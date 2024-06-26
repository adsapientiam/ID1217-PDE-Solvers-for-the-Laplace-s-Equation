Parallel Multigrid solver for laplace equation in 2D

## Src
[parallelmultigrid.c](parallelmultigrid.c)

## Compilation
```bash
gcc -O2 -o parallelmultigrid parallelmultigrid.c -lm -fopenmp
```

## Run

Parameters:
- grid_size: size of the grid
- num_iterations: number of iterations
- num_threads: number of threads to use

Example:
```bash
./parallelmultigrid 200 1000 4 
```