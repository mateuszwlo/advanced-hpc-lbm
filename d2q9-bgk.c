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

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

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

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, int* obstacles);
float collision(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, 
  float* tspeed0, float* tspeed1, float* tspeed2, float* tspeed3, float* tspeed4, float* tspeed5,  float* tspeed6, float* tspeed7, float* tspeed8, int* obstacles);
int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8);

/* compute average velocity */
float av_velocity(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */

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

  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

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
  initialise(paramfile, obstaclefile, &params, 
  &speed0, &speed1, &speed2, &speed3, &speed4, &speed5,  &speed6, &speed7, &speed8, 
  &tspeed0, &tspeed1, &tspeed2, &tspeed3, &tspeed4, &tspeed5,  &tspeed6, &tspeed7, &tspeed8, 
  &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

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

    accelerate_flow(params, speed0, speed1, speed2, speed3, speed4, speed5,  speed6, speed7, speed8, obstacles);
    av_vels[tt] = collision(params, speed0, speed1, speed2, speed3, speed4, speed5,  speed6, speed7, speed8, tspeed0, tspeed1, tspeed2, tspeed3, tspeed4, tspeed5,  tspeed6, tspeed7, tspeed8, obstacles);

    float* tmp = speed0;
    speed0 = tspeed0;
    tspeed0 = tmp;

    tmp = speed1;
    speed1 = tspeed1;
    tspeed1 = tmp;

    tmp = speed2;
    speed2 = tspeed2;
    tspeed2 = tmp;

    tmp = speed3;
    speed3 = tspeed3;
    tspeed3 = tmp;

    tmp = speed4;
    speed4 = tspeed4;
    tspeed4 = tmp;

    tmp = speed5;
    speed5 = tspeed5;
    tspeed5 = tmp;

    tmp = speed6;
    speed6 = tspeed6;
    tspeed6 = tmp;

    tmp = speed7;
    speed7 = tspeed7;
    tspeed7 = tmp;

    tmp = speed8;
    speed8 = tspeed8;
    tspeed8 = tmp;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  float reynolds = av_vels[params.maxIters - 1] * params.reynolds_dim / (1.f / 6.f * (2.f / params.omega - 1.f));
  printf("Reynolds number:\t\t%.12E\n", reynolds);
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles, av_vels);
  finalise(&params, &speed0, &speed1, &speed2, &speed3, &speed4, &speed5, &speed6, &speed7, &speed8, &tspeed0, &tspeed1, &tspeed2, &tspeed3, &tspeed4, &tspeed5, &tspeed6, &tspeed7, &tspeed8, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, float* restrict speed0, float* restrict speed1, float* restrict speed2, float* restrict speed3, float* restrict speed4, float* restrict speed5,  
                    float* restrict speed6, float* restrict speed7, float* restrict speed8, int* restrict obstacles)
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
  __assume_aligned(obstacles, 64);  


  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;
  for (int ii = 0; ii < params.nx; ii++)
  {
    int index = ii + jj*params.nx;
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (speed3[index] - w1) > 0.f
        && (speed6[index] - w2) > 0.f
        && (speed7[index] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      speed1[index] += w1;
      speed5[index] += w2;
      speed8[index] += w2;
      /* decrease 'west-side' densities */
      speed3[index] -= w1;
      speed6[index] -= w2;
      speed7[index] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float collision(const t_param params, float* restrict speed0, float* restrict speed1, float* restrict speed2, float* restrict speed3, float* restrict speed4, float* restrict speed5, 
                float* restrict speed6, float* restrict speed7, float* restrict speed8, float* restrict tspeed0, float* restrict tspeed1, float* restrict tspeed2, float* restrict tspeed3, 
                float* restrict tspeed4, float* restrict tspeed5,  float* restrict tspeed6, float* restrict tspeed7, float* restrict tspeed8, int* restrict obstacles)
{
  int tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;  /* accumulated magnitudes of velocity for each cell */

  __assume(params.nx % 2 == 0);
  __assume(params.nx % 4 == 0);
  __assume(params.nx % 8 == 0);
  __assume(params.nx % 16 == 0);
  __assume(params.nx % 32 == 0);
  __assume(params.nx % 64 == 0);
  __assume(params.nx % 128 == 0);
  __assume(params.nx % 256 == 0);
  __assume(params.ny % 2 == 0);
  __assume(params.ny % 4 == 0);
  __assume(params.ny % 8 == 0);
  __assume(params.ny % 16 == 0);
  __assume(params.ny % 32 == 0);
  __assume(params.ny % 64 == 0);
  __assume(params.ny % 128 == 0);
  __assume(params.ny % 256 == 0);

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

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma vector aligned 
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int index = ii + jj*params.nx;
      int y_n = jj + 1;
      if(y_n == params.ny) y_n = 0;
      int x_e = ii + 1;
      if(x_e == params.nx) x_e = 0;
      int y_s = jj - 1;
      if(y_s == -1) y_s = params.ny - 1;
      int x_w = ii - 1;
      if(x_w == -1) x_w = params.nx - 1;

      const float s0 = speed0[ii + jj*params.nx]; /* central cell, no movement */
      const float s1 = speed1[x_w + jj*params.nx]; /* east */
      const float s2 = speed2[ii + y_s*params.nx]; /* north */
      const float s3 = speed3[x_e + jj*params.nx]; /* west */
      const float s4 = speed4[ii + y_n*params.nx]; /* south */
      const float s5 = speed5[x_w + y_s*params.nx]; /* north-east */
      const float s6 = speed6[x_e + y_s*params.nx]; /* north-west */
      const float s7 = speed7[x_e + y_n*params.nx]; /* south-west */
      const float s8 = speed8[x_w + y_n*params.nx]; /* south-east */
      /* don't consider occupied cells */
      // if (obstacles[index])
      // {
      //   tspeed1[index] = s3;
      //   tspeed2[index] = s4;
      //   tspeed3[index] = s1;
      //   tspeed4[index] = s2;
      //   tspeed5[index] = s7;
      //   tspeed6[index] = s8;
      //   tspeed7[index] = s5;
      //   tspeed8[index] = s6;
      // }
      // else{
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

        /* zero velocity density: weight w0 */
        const float d0 = 4 * local_density * neg / 9;
        /* axis speeds: weight w1 */
        const float d1 = local_density * ((u_x * u_x / 2) + u_x + neg) / 9;
        const float d2 = local_density * ((u_y * u_y / 2) + u_y + neg) / 9;
        const float d3 = local_density * ((u_x * u_x / 2) - u_x + neg) / 9;
        const float d4 = local_density * ((u_y * u_y / 2) - u_y + neg) / 9;
        /* diagonal speeds: weight w2 */

        const float d5 = local_density * ((u_xy * u_xy / 2) + u_xy + neg) / 36;
        const float d6 = local_density * ((u_yx * u_yx / 2) + u_yx + neg) / 36;
        const float d7 = local_density * ((u_xy * u_xy / 2) - u_xy + neg) / 36;
        const float d8 = local_density * ((u_yx * u_yx / 2) - u_yx + neg) / 36;

        tspeed0[index] = s0 * (1 - params.omega) + (params.omega * d0);
        tspeed1[index] = s1 * (1 - params.omega) + (params.omega * d1);
        tspeed2[index] = s2 * (1 - params.omega) + (params.omega * d2);
        tspeed3[index] = s3 * (1 - params.omega) + (params.omega * d3);
        tspeed4[index] = s4 * (1 - params.omega) + (params.omega * d4);
        tspeed5[index] = s5 * (1 - params.omega) + (params.omega * d5);
        tspeed6[index] = s6 * (1 - params.omega) + (params.omega * d6);
        tspeed7[index] = s7 * (1 - params.omega) + (params.omega * d7);
        tspeed8[index] = s8 * (1 - params.omega) + (params.omega * d8);

      if (obstacles[index])
      {
        tspeed1[index] = s3;
        tspeed2[index] = s4;
        tspeed3[index] = s1;
        tspeed4[index] = s2;
        tspeed5[index] = s7;
        tspeed6[index] = s8;
        tspeed7[index] = s5;
        tspeed8[index] = s6;
      }
      else{
        /* compute x velocity component */
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
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
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

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *speed0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *speed8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  //if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tspeed0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  *tspeed8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  //if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*speed0)[ii + jj*params->nx] = w0;
      /* axis directions */
      (*speed1)[ii + jj*params->nx] = w1;
      (*speed2)[ii + jj*params->nx] = w1;
      (*speed3)[ii + jj*params->nx] = w1;
      (*speed4)[ii + jj*params->nx] = w1;
      /* diagonals */
      (*speed5)[ii + jj*params->nx] = w2;
      (*speed6)[ii + jj*params->nx] = w2;
      (*speed7)[ii + jj*params->nx] = w2;
      (*speed8)[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
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
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** speed0, float** speed1, float** speed2, float** speed3, float** speed4, float** speed5,  float** speed6, float** speed7, float** speed8, 
  float** tspeed0, float** tspeed1, float** tspeed2, float** tspeed3, float** tspeed4, float** tspeed5,  float** tspeed6, float** tspeed7, float** tspeed8,
             int** obstacles_ptr, float** av_vels_ptr)
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

  return EXIT_SUCCESS;
}


// float calc_reynolds(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8, int* obstacles)
// {
//   const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

//   return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
// }

float total_density(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5,  float* speed6, float* speed7, float* speed8)
{
  float total = 0.f;  /* accumulator */

  __assume_aligned(speed0, 64);
  __assume_aligned(speed1, 64);
  __assume_aligned(speed2, 64);
  __assume_aligned(speed3, 64);
  __assume_aligned(speed4, 64);
  __assume_aligned(speed5, 64);
  __assume_aligned(speed6, 64);
  __assume_aligned(speed7, 64);
  __assume_aligned(speed8, 64);

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii + jj*params.nx;

      total += speed0[index] 
              + speed1[index] 
              + speed2[index] 
              + speed3[index] 
              + speed4[index] 
              + speed5[index] 
              + speed6[index]
              + speed7[index]
              + speed8[index];
  }
  }

  return total;
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
      if (obstacles[ii + jj*params.nx])
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
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
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