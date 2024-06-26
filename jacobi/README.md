Sequential Jacobi solver for laplace equation in 2D

## Src
[jacobi.c](jacobi.c)

## Compilation
```bash
gcc -O2 -o jacobi jacobi.c -lm
```

## Run

Parameters:
- grid_size: size of the grid
- num_iterations: number of iterations
- num_threads: number of threads to use

Example:
```bash
./jacobi 200 100 4 
```