/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define TOT_U_TAG 100
#define NX_TAG 101
#define NY_TAG 102
#define MAX_ITERS_TAG 103
#define REYNOLDS_DIM_TAG 104
#define DENSITY_TAG 105
#define ACCEL_TAG 106
#define OMEGA_TAG 107
#define OBSTACLES_TAG 108

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
               int** obstacles_ptr, float** av_vels_ptr, float** send_array_top, float** send_array_bottom, float** recv_array_top, float** recv_array_bottom, int rank, int nprocs);
/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, int* obstacles, int rank, int start, int end);
float collision(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, 
  float* tspeed0, float* tspeed1, float* tspeed2, float* tspeed3, float* tspeed4, float* tspeed5,  float* tspeed6, float* tspeed7, float* tspeed8, int* obstacles, int rank, int start, int end, int tt);
int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels);
int get_rank_start(int total, int rank, int nprocs);
int get_rank_end(int total, int rank, int nprocs);
int swap_pointers(float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, float** tspeed0, float** tspeed1, 
                  float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
             int** obstacles_ptr, float** av_vels_ptr, float** send_array_top, float** send_array_bottom, float** recv_array_top, float** recv_array_bottom);
/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char* paramfile = NULL;    /* name of the input parameter file */
  char* obstaclefile = NULL; /* name of a the input obstacle file */
  t_param params;              /* struct to hold parameter values */

  float* speed0 = NULL;
  float* speed1 = NULL;
  float* speed2 = NULL;
  float* speed3 = NULL;
  float* speed4 = NULL;
  float* speed5 = NULL;
  float* speed6 = NULL;
  float* speed7 = NULL;
  float* speed8 = NULL;

  float* tspeed0 = NULL;
  float* tspeed1 = NULL;
  float* tspeed2 = NULL;
  float* tspeed3 = NULL;
  float* tspeed4 = NULL;
  float* tspeed5 = NULL;
  float* tspeed6 = NULL;
  float* tspeed7 = NULL;
  float* tspeed8 = NULL;

  float* send_array_top = NULL;
  float* send_array_bottom = NULL;
  float* recv_array_top = NULL;
  float* recv_array_bottom = NULL;

  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  int nprocs, rank, size, ssize, start, end, tot_cells, length, halo_region_size;
  MPI_Status status;
  double compute_time = 0;
  double collate_time = 0;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  tot_cells = initialise(paramfile, obstaclefile, &params, 
  &speed0, &speed1, &speed2, &speed3, &speed4, &speed5,  &speed6, &speed7, &speed8, 
  &tspeed0, &tspeed1, &tspeed2, &tspeed3, &tspeed4, &tspeed5,  &tspeed6, &tspeed7, &tspeed8, 
  &obstacles, &av_vels, &send_array_top, &send_array_bottom, &recv_array_top, &recv_array_bottom, rank, nprocs);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  start = get_rank_start(params.ny, rank, nprocs);
  end = get_rank_end(params.ny, rank, nprocs);
  size = end - start;
  length = size + 2;
  ssize = size * params.nx;
  halo_region_size = NSPEEDS * params.nx;

  //Node n - 1
  int top = rank - 1;
  if(top < 0) top = nprocs - 1;
  //Node n + 1
  int bottom = rank + 1;
  if(bottom >= nprocs) bottom = 0;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    __assume_aligned(speed0, 64);
    __assume_aligned(speed1, 64);
    __assume_aligned(speed2, 64);
    __assume_aligned(speed3, 64);
    __assume_aligned(speed4, 64);
    __assume_aligned(speed5, 64);
    __assume_aligned(speed6, 64);
    __assume_aligned(speed7, 64);
    __assume_aligned(speed8, 64);
    __assume_aligned(tspeed0, 64);
    __assume_aligned(tspeed1, 64);
    __assume_aligned(tspeed2, 64);
    __assume_aligned(tspeed3, 64);
    __assume_aligned(tspeed4, 64);
    __assume_aligned(tspeed5, 64);
    __assume_aligned(tspeed6, 64);
    __assume_aligned(tspeed7, 64);
    __assume_aligned(tspeed8, 64);
    __assume_aligned(obstacles, 64);

    accelerate_flow(params, speed0, speed1, speed2, speed3, speed4, speed5,  speed6, speed7, speed8, obstacles, rank, start, end);
    float tot_u = collision(params, speed0, speed1, speed2, speed3, speed4, speed5,  speed6, speed7, speed8, tspeed0, tspeed1, tspeed2, tspeed3, tspeed4, tspeed5, tspeed6, tspeed7, tspeed8, obstacles, rank, start, end, tt);

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;
    compute_time += (comp_toc - comp_tic);

    //Swap pointers
    swap_pointers(&speed0, &speed1, &speed2, &speed3, &speed4, &speed5, &speed6, &speed7, &speed8, &tspeed0, &tspeed1, &tspeed2, &tspeed3, &tspeed4, &tspeed5, &tspeed6, &tspeed7, &tspeed8);

    //Sum all tot_u values and calculate average
    float result;
    MPI_Reduce(&tot_u, &result, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) av_vels[tt] = result / (float) tot_cells;
    
    //Sending top halo region
    #pragma vector aligned 
    for(int ii = 0; ii < params.nx; ii++){
      send_array_top[ii + 0*params.nx] = speed0[ii + params.nx];
      send_array_top[ii + 1*params.nx] = speed1[ii + params.nx];
      send_array_top[ii + 2*params.nx] = speed2[ii + params.nx];
      send_array_top[ii + 3*params.nx] = speed3[ii + params.nx];
      send_array_top[ii + 4*params.nx] = speed4[ii + params.nx];
      send_array_top[ii + 5*params.nx] = speed5[ii + params.nx];
      send_array_top[ii + 6*params.nx] = speed6[ii + params.nx];
      send_array_top[ii + 7*params.nx] = speed7[ii + params.nx];
      send_array_top[ii + 8*params.nx] = speed8[ii + params.nx];
    }

    #pragma vector aligned 
    for(int ii = 0; ii < params.nx; ii++){
      send_array_bottom[ii + 0*params.nx] = speed0[ii + (size * params.nx)];
      send_array_bottom[ii + 1*params.nx] = speed1[ii + (size * params.nx)];
      send_array_bottom[ii + 2*params.nx] = speed2[ii + (size * params.nx)];
      send_array_bottom[ii + 3*params.nx] = speed3[ii + (size * params.nx)];
      send_array_bottom[ii + 4*params.nx] = speed4[ii + (size * params.nx)];
      send_array_bottom[ii + 5*params.nx] = speed5[ii + (size * params.nx)];
      send_array_bottom[ii + 6*params.nx] = speed6[ii + (size * params.nx)];
      send_array_bottom[ii + 7*params.nx] = speed7[ii + (size * params.nx)];
      send_array_bottom[ii + 8*params.nx] = speed8[ii + (size * params.nx)];
    }

    MPI_Sendrecv(send_array_top, halo_region_size, MPI_FLOAT, top, 0, recv_array_top, halo_region_size, MPI_FLOAT, bottom, 0, MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_array_bottom, halo_region_size, MPI_FLOAT, bottom, 0, recv_array_bottom, halo_region_size, MPI_FLOAT, top, 0, MPI_COMM_WORLD, &status);

    #pragma vector aligned 
    for(int ii = 0; ii < params.nx; ii++){
      speed0[ii] = recv_array_bottom[ii + 0*params.nx];
      speed1[ii] = recv_array_bottom[ii + 1*params.nx];
      speed2[ii] = recv_array_bottom[ii + 2*params.nx];
      speed3[ii] = recv_array_bottom[ii + 3*params.nx];
      speed4[ii] = recv_array_bottom[ii + 4*params.nx];
      speed5[ii] = recv_array_bottom[ii + 5*params.nx];
      speed6[ii] = recv_array_bottom[ii + 6*params.nx];
      speed7[ii] = recv_array_bottom[ii + 7*params.nx];
      speed8[ii] = recv_array_bottom[ii + 8*params.nx];
    }

    #pragma vector aligned 
    for(int ii = 0; ii < params.nx; ii++){
      speed0[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 0*params.nx];
      speed1[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 1*params.nx];
      speed2[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 2*params.nx];
      speed3[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 3*params.nx];
      speed4[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 4*params.nx];
      speed5[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 5*params.nx];
      speed6[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 6*params.nx];
      speed7[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 7*params.nx];
      speed8[ii + ((size + 1) * params.nx)] = recv_array_top[ii + 8*params.nx];   
    }
    /* Collate time stops here, compute time starts again*/
    if(tt != params.maxIters - 1){
      gettimeofday(&timstr, NULL);
      col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
      comp_tic=col_toc;
      collate_time += (col_toc - col_tic);
    }
  }

  if(rank == 0){
    float* final_speed0 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed1 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed2 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed3 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed4 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed5 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed6 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed7 = malloc(sizeof(float) * params.ny * params.nx);
    float* final_speed8 = malloc(sizeof(float) * params.ny * params.nx);

    //Copy itself's chunk
    for(int jj = 0; jj < size; jj++){
      for(int ii = 0; ii < params.nx; ii++){
        int index = ii + jj*params.nx;
        final_speed0[index] = speed0[index + params.nx];
        final_speed1[index] = speed1[index + params.nx];
        final_speed2[index] = speed2[index + params.nx];
        final_speed3[index] = speed3[index + params.nx];
        final_speed4[index] = speed4[index + params.nx];
        final_speed5[index] = speed5[index + params.nx];
        final_speed6[index] = speed6[index + params.nx];
        final_speed7[index] = speed7[index + params.nx];
        final_speed8[index] = speed8[index + params.nx];
      }
    }

    for(int r = 1; r < nprocs; r++){
      int s = get_rank_start(params.ny, r, nprocs);
      int e = get_rank_end(params.ny, r, nprocs);
      int incomingSize = e - s;
      int incomingArraySize = incomingSize * params.nx;

      float* array = malloc(sizeof(float) * incomingArraySize * NSPEEDS);

      MPI_Recv(array, incomingArraySize * NSPEEDS, MPI_FLOAT, r, 20, MPI_COMM_WORLD, &status);
      for(int jj = 0; jj < incomingSize; jj++){
        for(int ii = 0; ii < params.nx; ii++){
          int index = ii + jj*params.nx;
          int y_n = ii + ((s + jj) * params.nx);
          final_speed0[y_n] = array[ii + ((jj + (0 * incomingSize)) * params.nx)];
          final_speed1[y_n] = array[ii + ((jj + (1 * incomingSize)) * params.nx)];
          final_speed2[y_n] = array[ii + ((jj + (2 * incomingSize)) * params.nx)];
          final_speed3[y_n] = array[ii + ((jj + (3 * incomingSize)) * params.nx)];
          final_speed4[y_n] = array[ii + ((jj + (4 * incomingSize)) * params.nx)];
          final_speed5[y_n] = array[ii + ((jj + (5 * incomingSize)) * params.nx)];
          final_speed6[y_n] = array[ii + ((jj + (6 * incomingSize)) * params.nx)];
          final_speed7[y_n] = array[ii + ((jj + (7 * incomingSize)) * params.nx)];
          final_speed8[y_n] = array[ii + ((jj + (8 * incomingSize)) * params.nx)];
        }
      }

      free(array);
    }

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;
    collate_time += (col_toc - col_tic);

    /* write final values and free memory */
    write_values(params, final_speed0, final_speed1, final_speed2, final_speed3, final_speed4, final_speed5, final_speed6, final_speed7, final_speed8, obstacles, av_vels);
    free(final_speed0);
    free(final_speed1);
    free(final_speed2);
    free(final_speed3);
    free(final_speed4);
    free(final_speed5);
    free(final_speed6);
    free(final_speed7);
    free(final_speed8);

    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", av_vels[params.maxIters - 1] * params.reynolds_dim / (1.f / 6.f * (2.f / params.omega - 1.f)));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", compute_time);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", collate_time);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  else{
    float* array = malloc(sizeof(float) * NSPEEDS * ssize);

    for(int jj = 0; jj < size; jj++){
      for(int ii = 0; ii < params.nx; ii++){
        int index = ii + jj*params.nx;
        array[ii + ((jj + (0 * size)) * params.nx)] = speed0[index + params.nx];
        array[ii + ((jj + (1 * size)) * params.nx)] = speed1[index + params.nx];
        array[ii + ((jj + (2 * size)) * params.nx)] = speed2[index + params.nx];
        array[ii + ((jj + (3 * size)) * params.nx)] = speed3[index + params.nx];
        array[ii + ((jj + (4 * size)) * params.nx)] = speed4[index + params.nx];
        array[ii + ((jj + (5 * size)) * params.nx)] = speed5[index + params.nx];
        array[ii + ((jj + (6 * size)) * params.nx)] = speed6[index + params.nx];
        array[ii + ((jj + (7 * size)) * params.nx)] = speed7[index + params.nx];
        array[ii + ((jj + (8 * size)) * params.nx)] = speed8[index + params.nx];
      }
    }

    MPI_Send(array, NSPEEDS * ssize, MPI_FLOAT, 0, 20, MPI_COMM_WORLD);
    free(array);

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;
    collate_time += (col_toc - col_tic);
    
    /* write final values and free memory */
    printf("==done==\n");
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", compute_time);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", collate_time);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  finalise(&params, &speed0, &speed1, &speed2, &speed3, &speed4, &speed5, &speed6, &speed7, &speed8, &tspeed0, &tspeed1, 
           &tspeed2, &tspeed3, &tspeed4, &tspeed5, &tspeed6, &tspeed7, &tspeed8, &obstacles, &av_vels, &send_array_top, &send_array_bottom, &recv_array_top, &recv_array_bottom);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

int swap_pointers(float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, float** tspeed0, float** tspeed1, 
                  float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8){
  float* tmp = *speed0;
    *speed0 = *tspeed0;
    *tspeed0 = tmp;

    tmp = *speed1;
    *speed1 = *tspeed1;
    *tspeed1 = tmp;

    tmp = *speed2;
    *speed2 = *tspeed2;
    *tspeed2 = tmp;

    tmp = *speed3;
    *speed3 = *tspeed3;
    *tspeed3 = tmp;

    tmp = *speed4;
    *speed4 = *tspeed4;
    *tspeed4 = tmp;

    tmp = *speed5;
    *speed5 = *tspeed5;
    *tspeed5 = tmp;

    tmp = *speed6;
    *speed6 = *tspeed6;
    *tspeed6 = tmp;

    tmp = *speed7;
    *speed7 = *tspeed7;
    *tspeed7 = tmp;

    tmp = *speed8;
    *speed8 = *tspeed8;
    *tspeed8 = tmp;

    return EXIT_SUCCESS;
  }

int get_rank_start(int total, int rank, int nprocs){
  int size = total / nprocs;
  int remainder = total % nprocs;

  if(rank < remainder){
    return (rank * size) + rank;
  }
  else{
    return (rank * size) + remainder;
  }
}

int get_rank_end(int total, int rank, int nprocs){
  int size = total / nprocs;
  int remainder = total % nprocs;

  if(rank < remainder){
    int start = (rank * size) + rank;
    return start + size + 1;
  }
  else{
    int start = (rank * size) + remainder;
    return start + size;
  }
}

int accelerate_flow(const t_param params, float* restrict speed0, float* restrict speed1, float* restrict speed2, float* restrict speed3, float* restrict speed4, float* restrict speed5,  
                    float* restrict speed6, float* restrict speed7, float* restrict speed8, int* restrict obstacles, int rank, int start, int end)
{
  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  int index; 

  //If halo exchange edge is 2nd row of the grid
  if(start - 1 == jj) index = 0;
  else if(end == jj) index = (end - start + 1) * params.nx;
  else if(start <= jj && end > jj) index = (jj - start + 1) * params.nx;
  else return EXIT_SUCCESS;

  __assume_aligned(speed0, 64);
  __assume_aligned(speed1, 64);
  __assume_aligned(speed2, 64);
  __assume_aligned(speed3, 64);
  __assume_aligned(speed4, 64);
  __assume_aligned(speed5, 64);
  __assume_aligned(speed6, 64);
  __assume_aligned(speed7, 64);
  __assume_aligned(speed8, 64);
  __assume_aligned(obstacles, 64);  

  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  for (int i = 0; i < params.nx; i++)
  {
    int ii = i + index;
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[i + jj*params.nx]
        && (speed3[ii] - w1) > 0.f
        && (speed6[ii] - w2) > 0.f
        && (speed7[ii] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      speed1[ii] += w1;
      speed5[ii] += w2;
      speed8[ii] += w2;
      /* decrease 'west-side' densities */
      speed3[ii] -= w1;
      speed6[ii] -= w2;
      speed7[ii] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float collision(const t_param params, float* restrict speed0, float* restrict speed1, float* restrict speed2, float* restrict speed3, float* restrict speed4, float* restrict speed5, 
                float* restrict speed6, float* restrict speed7, float* restrict speed8, float* restrict tspeed0, float* restrict tspeed1, float* restrict tspeed2, float* restrict tspeed3, 
                float* restrict tspeed4, float* restrict tspeed5,  float* restrict tspeed6, float* restrict tspeed7, float* restrict tspeed8, int* restrict obstacles, int rank, int start, int end, int tt)
{
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  float tot_u = 0.f;
  int size = end - start;

  for (int jj = 1; jj < size + 1; jj++)
  {
    #pragma vector aligned 
    for (int ii = 0; ii < params.nx; ii++)
    {
      int y_n = jj + 1;
      int y_s = jj - 1;

      int x_e = ii + 1;
      if(x_e == params.nx) x_e = 0;
      int x_w = ii - 1;
      if(x_w == -1) x_w = params.nx - 1;

      const int index = ii + jj*params.nx;

      const float s0 = speed0[ii + jj*params.nx]; /* central cell, no movement */
      const float s1 = speed1[x_w + jj*params.nx]; /* east */
      const float s2 = speed2[ii + y_s*params.nx]; /* north */
      const float s3 = speed3[x_e + jj*params.nx]; /* west */
      const float s4 = speed4[ii + y_n*params.nx]; /* south */
      const float s5 = speed5[x_w + y_s*params.nx]; /* north-east */
      const float s6 = speed6[x_e + y_s*params.nx]; /* north-west */
      const float s7 = speed7[x_e + y_n*params.nx]; /* south-west */
      const float s8 = speed8[x_w + y_n*params.nx]; /* south-east */

      tspeed1[index] = s3;
      tspeed2[index] = s4;
      tspeed3[index] = s1;
      tspeed4[index] = s2;
      tspeed5[index] = s7;
      tspeed6[index] = s8;
      tspeed7[index] = s5;
      tspeed8[index] = s6;

      int abs_index = ii + ((start + jj - 1) * params.nx); 
      if(!obstacles[abs_index]){
        /* compute local density total */
        const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;

        /* compute x velocity component */
        const float u_x = (s1 + s5 + s8 - (s3 + s6  + s7)) / local_density * 3;
        /* compute y velocity component */
        const float u_y = (s2 + s5 + s6 - (s4  + s7 + s8)) / local_density * 3;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;
        const float u_xy = u_x + u_y;
        const float u_yx = u_y - u_x;
        const float neg = 1 - (u_sq / 6);

        tspeed0[index] = s0 * (1 - params.omega) + (params.omega * 4 * local_density * neg / 9);
        tspeed1[index] = s1 * (1 - params.omega) + (params.omega * local_density * ((u_x * u_x / 2) + u_x + neg) / 9);
        tspeed2[index] = s2 * (1 - params.omega) + (params.omega * local_density * ((u_y * u_y / 2) + u_y + neg) / 9);
        tspeed3[index] = s3 * (1 - params.omega) + (params.omega * local_density * ((u_x * u_x / 2) - u_x + neg) / 9);
        tspeed4[index] = s4 * (1 - params.omega) + (params.omega * local_density * ((u_y * u_y / 2) - u_y + neg) / 9);
        tspeed5[index] = s5 * (1 - params.omega) + (params.omega * local_density * ((u_xy * u_xy / 2) + u_xy + neg) / 36);
        tspeed6[index] = s6 * (1 - params.omega) + (params.omega * local_density * ((u_yx * u_yx / 2) + u_yx + neg) / 36);
        tspeed7[index] = s7 * (1 - params.omega) + (params.omega * local_density * ((u_xy * u_xy / 2) - u_xy + neg) / 36);
        tspeed8[index] = s8 * (1 - params.omega) + (params.omega * local_density * ((u_yx * u_yx / 2) - u_yx + neg) / 36);
        
        /* x-component of velocity */
        const float tu_x = (tspeed1[index]
                + tspeed5[index]
                + tspeed8[index]
                - (tspeed3[index]
                    + tspeed6[index]
                    + tspeed7[index]))
                / local_density;

          /* compute y velocity component */
          const float tu_y = (tspeed2[index]
                + tspeed5[index]
                + tspeed6[index]
                - (tspeed4[index]
                    + tspeed7[index]
                    + tspeed8[index]))
                / local_density;
          
          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrtf((tu_x * tu_x) + (tu_y * tu_y));
      }
    }
  }
  return tot_u;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
               int** obstacles_ptr, float** av_vels_ptr, float** send_array_top, float** send_array_bottom, float** recv_array_top, float** recv_array_bottom, int rank, int nprocs)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  int vals1[4];
  float vals2[3];

  /* open the parameter file */
  if(rank == 0){
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
      sprintf(message, "could not open input parameter file: %s", paramfile);
      die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->density));

    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->accel));

    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->omega));

    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    vals1[0] = params->nx;
    vals1[1] = params->ny;
    vals1[2] = params->maxIters;
    vals1[3] = params->reynolds_dim;

    vals2[0] = params->density;
    vals2[1] = params->accel;
    vals2[2] = params->omega;
  }

  MPI_Bcast(vals1, 4, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vals2, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if(rank != 0){
    params->nx = vals1[0];
    params->ny = vals1[1];
    params->maxIters = vals1[2];
    params->reynolds_dim = vals1[3];

    params->density = vals2[0];
    params->accel = vals2[1];
    params->omega = vals2[2];
  }

  int start = get_rank_start(params->ny, rank, nprocs);
  int end = get_rank_end(params->ny, rank, nprocs);
  int size = end - start;
  int ARRAY_SIZE = (size + 2) * params->nx;
  int tot_cells = params->nx * params->ny;
  int halo_region_size = params->nx * NSPEEDS;

  *send_array_top = _mm_malloc(sizeof(float) * halo_region_size, 64);
  *send_array_bottom = _mm_malloc(sizeof(float) * halo_region_size, 64);
  *recv_array_top = _mm_malloc(sizeof(float) * halo_region_size, 64);  
  *recv_array_bottom = _mm_malloc(sizeof(float) * halo_region_size, 64); 

  /* main grid */
  *speed0 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed1 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed2 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed3 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed4 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed5 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed6 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed7 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *speed8 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);

  /* 'helper' grid, used as scratch space */
  *tspeed0 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed1 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed2 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed3 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed4 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed5 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed6 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed7 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);
  *tspeed8 = _mm_malloc(sizeof(float) * ARRAY_SIZE, 64);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  /* first set all cells in obstacle array to zero */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int i = 0; i < ARRAY_SIZE; i++){
      /* centre */
      (*speed0)[i] = w0;
      /* axis directions */
      (*speed1)[i] = w1;
      (*speed2)[i] = w1;
      (*speed3)[i] = w1;
      (*speed4)[i] = w1;
      /* diagonals */
      (*speed5)[i] = w2;
      (*speed6)[i] = w2;
      (*speed7)[i] = w2;
      (*speed8)[i] = w2;
  }

  if(rank == 0){
    for(int jj = 0; jj < params->ny; jj++){
      for(int ii = 0; ii < params->nx; ii++){
        (*obstacles_ptr)[ii + jj*params->nx] = 0;
      }
    }
    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
      sprintf(message, "could not open input obstacles file: %s", obstaclefile);
      die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
      /* some checks */
      if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

      if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

      if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

      if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

      /* assign to array */
      (*obstacles_ptr)[xx + yy*params->nx] = blocked;
      --tot_cells;
    }

    /* and close the file */
    fclose(fp);
  }

  MPI_Bcast(*obstacles_ptr, params->ny * params->nx, MPI_INT, 0, MPI_COMM_WORLD);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);

  return tot_cells;
}

int finalise(const t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
             int** obstacles_ptr, float** av_vels_ptr, float** send_array_top, float** send_array_bottom, float** recv_array_top, float** recv_array_bottom)
{
  /*
  ** free up allocated memory
  */
  _mm_free(*speed0);
  *speed0 = NULL;

  _mm_free(*speed1);
  *speed1 = NULL;

  _mm_free(*speed2);
  *speed2 = NULL;

  _mm_free(*speed3);
  *speed3 = NULL;

  _mm_free(*speed4);
  *speed4 = NULL;

  _mm_free(*speed5);
  *speed5 = NULL;

  _mm_free(*speed6);
  *speed6 = NULL;

  _mm_free(*speed7);
  *speed7 = NULL; 

  _mm_free(*speed8);
  *speed8 = NULL;

  _mm_free(*tspeed0);
  *tspeed0 = NULL;

  _mm_free(*tspeed1);
  *tspeed1 = NULL;

  _mm_free(*tspeed2);
  *tspeed2 = NULL;

  _mm_free(*tspeed3);
  *tspeed3 = NULL;

  _mm_free(*tspeed4);
  *tspeed4 = NULL;

  _mm_free(*tspeed5);
  *tspeed5 = NULL;

  _mm_free(*tspeed6);
  *tspeed6 = NULL;

  _mm_free(*tspeed7);
  *tspeed7 = NULL; 

  _mm_free(*tspeed8);
  *tspeed8 = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  _mm_free(*send_array_top);
  *send_array_top = NULL;

  _mm_free(*send_array_bottom);
  *send_array_bottom = NULL;

  _mm_free(*recv_array_top);
  *recv_array_top = NULL;

  _mm_free(*recv_array_bottom);
  *recv_array_bottom = NULL;

  return EXIT_SUCCESS;
}

int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  __assume_aligned(obstacles, 64);
  __assume_aligned(av_vels, 64);

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii + jj*params.nx;
      /* an occupied cell */
      if (obstacles[index])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = speed0[index] 
                      + speed1[index] 
                      + speed2[index] 
                      + speed3[index] 
                      + speed4[index] 
                      + speed5[index] 
                      + speed6[index]
                      + speed7[index]
                      + speed8[index];

        /* compute x velocity component */
        u_x = (speed1[index]
               + speed5[index]
               + speed8[index]
               - (speed3[index]
                  + speed6[index]
                  + speed7[index]))
              / local_density;
        /* compute y velocity component */
        u_y = (speed2[index]
               + speed5[index]
               + speed6[index]
               - (speed4[index]
                  + speed7[index]
                  + speed8[index]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[index]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}