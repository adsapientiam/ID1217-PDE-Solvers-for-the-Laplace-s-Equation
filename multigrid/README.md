Sequential Multigrid solver for laplace equation in 2D

## Src
[multigrid.c](multigrid.c)

## Compilation
```bash
gcc -O2 -o multigrid multigrid.c -lm
```

## Run

Parameters:
- grid_size: size of the grid
- num_iterations: number of iterations
- num_threads: number of threads to use

Example:
```bash
./multigrid 200 1000 4 
```