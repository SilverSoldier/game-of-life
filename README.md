## Parallel Conway's Game of Life

Parallel Game of Life written using Open MPI and OpenMP.

Each process works on a subset of the rows of the board and communicates the first and last row to the previous and next process respectively at every iteration using OpenMPI.

Within a process, OpenMP is used to progress game to the next step in parallel.
