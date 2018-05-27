#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define board(x, y, c) (board[((x) * (c)) + (y)])
#define live(i) ((board[(i)]) % 2)

void print_board(int* board, int rows, int columns){
  for(int i = 0; i < rows; i++){
	for(int j = 0; j < columns; j++)
	  printf("%d\n", board[i * columns + j]);
  }
}

/* Step 6 : Function to iterate through all the elements of the board and move to the next step */

/* Rules for progress to next step:
 * Live -> Live 1
 * Live -> Dead 3
 * Dead -> Dead 0
 * Dead -> Live 2
 * If value is 1 or 3 => live in current step.
 * If value is 0 or 2 => dead in current step.
 * If value is 2 or 3 => update  in next step.
 */
void move_next_step(int rows, int c, int* board){
#pragma omp parallel for
  for(int i = 0; i < rows * c; i++){
	/* Living cell with less than 2 live neighbors dies */
	/* Living cell with more than 3 live neighbors dies */
	/* Dead cell with exactly 3 live neighbors spawns */
	int live_nbr_count = 0;
	if(i % c != 0){
	  live_nbr_count += live(i - c - 1);
	  live_nbr_count += live(i - 1);
	  live_nbr_count += live(i + c - 1);
	}
	live_nbr_count += live(i - c);
	live_nbr_count += live(i + c);

	/* Doing test in the middle - in case it reduces total no. of operations */
	if(live_nbr_count > 3){
	  if(live(i))
		/* Live cell will definitely die */
#pragma omp atomic write
		board[i] = 3;
	  /* Dead cell definitely cannot live */
	  continue;
	}

	if((i + 1) % c != 0){
	  live_nbr_count += live(i - c + 1);
	  live_nbr_count += live(i + 1);
	  live_nbr_count += live(i + c + 1);
	}

	if(live(i)){
	  if(live_nbr_count != 3 && live_nbr_count != 2)
#pragma omp atomic write
		board[i] = 3;
	} else if(live_nbr_count == 3) {
#pragma omp atomic write
	  board[i] = 2;
	}
  }
}

int main(int argc, char *argv[]) {
  int seed, rows, columns, generations;
  char* filename = (char*) malloc(100 * sizeof(char));

  int n_threads, world_size, rank;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Check command line arguments */
  if (argc < 6) {
	if (rank == 0)
	  printf("Usage: %s [num threads] [Seed] [Rows] [Columns] [Generations] <filename>\n", argv[0]);
	exit (1);
  }

  sscanf(argv[1], "%d", &n_threads);
  omp_set_num_threads(n_threads);

  /* Parse the arguments */
  sscanf(argv[2], "%d", &seed);
  sscanf(argv[3], "%d", &rows);
  sscanf(argv[4], "%d", &columns);
  sscanf(argv[5], "%d", &generations);

  bool interchanged = false;

  /* Step 1: All processes intitalize entire board */
  int** init_board = (int**) calloc(rows, sizeof(int*));
  for(int i = 0; i < rows; i++)
	init_board[i] = (int*) calloc(columns, sizeof(int));

  if(argc == 7){
	sscanf(argv[6], "%s", filename);
	/* Read array from the file */
	FILE* fin = fopen(filename, "r");
	for(int i = 0; i < rows; i++)
	  for(int j = 0; j < columns; j++)
		fscanf(fin, "%d", &init_board[i][j]);
	fclose(fin);
  } else {
	srand(seed);
	for(int i = 0; i < rows; i++)
	  for(int j = 0; j < columns; j++)
		init_board[i][j] = rand() % 2;
  }

  free(filename);

  /* We would like the number of rows to be greater than the number of columns.
   * So, interchange if not true */
  if(columns > rows){
	int temp = columns;
	columns = rows;
	rows = temp;
	interchanged = true;
  }

  /* Step 2: all processes find their chunk size */
  /* Find row numbers for current process */
  /* Doing equal work as much as possible by dividing remainder over all the remaining processes */
  int row_start, row_end;
  int chunk_size = (rows/world_size);
  int rmdr = rows - chunk_size * world_size;
  row_start = chunk_size * rank;
  if(rank < rmdr){
	chunk_size++;
	row_start += rank;
  }
  else
	row_start += rmdr;

  row_end = row_start + chunk_size;

  /* Board is a row wise flattened 1D structure */
  int* board = (int*) calloc((chunk_size + 2) * columns, sizeof(int));
  int row_size = columns * sizeof(int);

  if(interchanged){
	int offset = row_start - (rank != 0);
	int i_range = chunk_size + (rank != 0) + (rank != world_size - 1);

	int k = (rank == 0) * columns;
	for(int i = 0; i < i_range; i++){
	  for(int j = 0; j < columns; j++){
		board[k++] = init_board[j][offset];
	  }
	  offset++;
	}
  } else {
	int src = row_start - (rank != 0);
	int dest = (rank == 0) * columns;
	int size = chunk_size + (rank != 0) + (rank != (world_size-1));
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
	  memcpy(board + dest + i*columns, init_board[src+i], row_size);
	}
  }

  free(init_board);

  MPI_Request request[2], send_req[2];
  MPI_Status recv_status[2];
  int flag0, flag1;

  for(int iter = 0; iter < generations; iter++){
	flag0 = flag1 = 1;
	/* Step 3 : Do non-blocking send and recv of previous and next rows except for the 1st iteration - all processes know everything */
	if(iter != 0){
	  if(rank != 0){
		flag0 = 0;
		/* Receive previous row information - non blocking */
		MPI_Irecv(board, columns, MPI_INT, rank - 1, 10, MPI_COMM_WORLD, &request[0]);
	  }
	  if(rank != world_size - 1){
		flag1 = 0;
		/* Receive next row information - non blocking */
		MPI_Irecv(board + (chunk_size + 1) * columns, columns, MPI_INT, rank + 1, 11, MPI_COMM_WORLD, &request[1]);
	  }
	  if(rank != 0){
		/* Send previous row information - non blocking*/
		MPI_Isend(board + columns, columns, MPI_INT, rank - 1, 11, MPI_COMM_WORLD, &send_req[1]);
	  }
	  if(rank != world_size - 1){
		/* Send next row information - non blocking*/
		MPI_Isend(board + chunk_size * columns, columns, MPI_INT, rank + 1, 10, MPI_COMM_WORLD, &send_req[0]);
	  }
	}

	/* Step 4 : Perform Game of Life for rows 2 to last but one */
	int start_location = (1 + (!flag0)) * columns;
	int size = chunk_size - (!flag0) - (!flag1);

	move_next_step(size, columns, board + start_location);

	/* Step 5 : Check if the previous and next rows have been received now and perform for whichever has been received */
	while(!flag0 || !flag1) {
	  if(!flag0){
		MPI_Test(&request[0], &flag0, &recv_status[0]);
		if(flag0){
		  move_next_step(1, columns, board + columns);
		}
	  }
	  if(!flag1){
		MPI_Test(&request[1], &flag1, &recv_status[1]);
		if(flag1){
		  move_next_step(1, columns, board + chunk_size * columns);
		}
	  }
	}
	/* Change all progress elements */
#pragma omp parallel for
	for(int i = columns; i < (chunk_size + 1) * columns; i++)
	  if(board[i] > 1)
		board[i] = 3 - board[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Step 7: Receive the final board from all (using Gatherv) and print it. */
  if(rank == 0) {
	int chunk_sizes[world_size];
	/* Initially set all chunks as same size */
	for(int i = 0; i < world_size; i++)
	  chunk_sizes[i] = (rows/world_size) * columns;
	/* Increment chunk_size by 1 for remaining processes */
	if(rmdr != 0)
	  for(int i = 0; i < rmdr; i++)
		chunk_sizes[i] += columns;

	/* Use gatherv to receive chunks from all processes */
	int displacements[world_size];
	displacements[0] = 0;
	for(int i = 1; i < world_size; i++)
	  displacements[i] = displacements[i-1] + chunk_sizes[i-1];

	int *final_board = (int*) calloc(rows * columns, sizeof(int));
	MPI_Gatherv(board + columns, chunk_size * columns, MPI_INT, final_board, chunk_sizes, displacements, MPI_INT, 0, MPI_COMM_WORLD);
	if(interchanged){
	  for(int i = 0; i < columns; i++){
		int offset = 0;
		for(int j = 0; j < rows; j++){
		  printf("%d\n", final_board[i + offset]);
		  offset += columns;
		}
	  }
	} else{
	  print_board(final_board, rows, columns);
	}
	free(final_board);
  }
  else{
	MPI_Gatherv(board + columns, chunk_size * columns, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
  }

  free(board);
  MPI_Finalize();
  return 0;
}
