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
  float* restrict speed0;
  float* restrict speed1;
  float* restrict speed2;
  float* restrict speed3;
  float* restrict speed4;
  float* restrict speed5;
  float* restrict speed6;
  float* restrict speed7;
  float* restrict speed8;
} speeds;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, speeds* cells_ptr, speeds* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, speeds* cells, int* obstacles);
float collision(const t_param params, speeds* cells, speeds* tmp_cells, int* obstacles);
int write_values(const t_param params, speeds* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, speeds* cells_ptr, speeds* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, speeds* cells);

/* compute average velocity */
float av_velocity(const t_param params, speeds* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float average);

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
  speeds cells;    /* grid containing fluid densities */
  speeds tmp_cells;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
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
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow(params, &cells, obstacles);
    av_vels[tt] = collision(params, &cells, &tmp_cells, obstacles);

    speeds* a = &cells;
    cells = tmp_cells;
    tmp_cells = *a;

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
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, av_vels[params.maxIters - 1]));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, &cells, obstacles, av_vels);
  //finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

void map(int ii, int jj, int nx, int ny, int* mappings){
  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
    
  mappings[0] = ii + jj*nx;
  mappings[1] = x_w + jj*nx;
  mappings[2] = ii + y_s*nx;
  mappings[3] = x_e + jj*nx;
  mappings[4] = ii + y_n*nx;
  mappings[5] = x_w + y_s*nx;
  mappings[6] = x_e + y_s*nx;
  mappings[7] = x_e + y_n*nx;
  mappings[8] = x_w + y_n*nx;
}

int accelerate_flow(const t_param params, speeds* cells, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && ( cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && ( cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && ( cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float collision(const t_param params, speeds* restrict cells, speeds* restrict result, int* restrict obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  int tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;  /* accumulated magnitudes of velocity for each cell */
  /* equilibrium densities */
  float d_equ[NSPEEDS];
  int mappings[NSPEEDS];

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      map(ii, jj, params.nx, params.ny, mappings);

      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx]) 
        {
        /* compute local density total */
        float local_density = cells->speed0[mappings[0]] 
                              + cells->speed1[mappings[1]]
                              + cells->speed2[mappings[2]]
                              + cells->speed3[mappings[3]]
                              + cells->speed4[mappings[4]]
                              + cells->speed5[mappings[5]]
                              + cells->speed6[mappings[6]]
                              + cells->speed7[mappings[7]]
                              + cells->speed8[mappings[8]];

        /* compute x velocity component */
        float u_x = (cells->speed1[mappings[1]]
                      + cells->speed5[mappings[5]]
                      + cells->speed8[mappings[8]]
                      - (cells->speed3[mappings[3]]
                         + cells->speed6[mappings[6]]
                         + cells->speed7[mappings[7]]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[mappings[2]]
                      + cells->speed5[mappings[5]]
                      + cells->speed6[mappings[6]]
                      - (cells->speed4[mappings[4]]
                         + cells->speed7[mappings[7]]
                         + cells->speed8[mappings[8]]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;
        float neg = u_sq / (2.f * c_sq);
        float c_sq2 = (2.f * c_sq * c_sq);

        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        float tmp = w1 * local_density * (1.f + ((u_x * u_x) / c_sq2) - neg);
        float tmp1 = w1 * local_density * u_x / c_sq;
        d_equ[1] = tmp + tmp1;
        d_equ[3] = tmp - tmp1;

        tmp = w1 * local_density * (1.f + ((u_y * u_y) / c_sq2) - neg);
        tmp1 = w1 * local_density * u_y / c_sq;
        d_equ[2] = tmp + tmp1;
        d_equ[4] = tmp - tmp1;
        /* diagonal speeds: weight w2 */
        tmp = w2 * local_density * (1.f + ((u_x + u_y) * (u_x + u_y)) / c_sq2- neg);
        tmp1 = w2 * local_density * (u_x + u_y) / c_sq;
        d_equ[5] = tmp + tmp1;
        d_equ[7] = tmp - tmp1;

        tmp = w2 * local_density * (1.f + ((- u_x + u_y) * (- u_x + u_y)) / c_sq2- neg);
        tmp1 = w2 * local_density *  (- u_x + u_y) / c_sq;
        d_equ[6] = tmp + tmp1;
        d_equ[8] = tmp - tmp1;

        /* local density total */
        float av_local_density = 0.f;

        /* relaxation step */
        float temp = cells->speed0[mappings[0]] + params.omega * (d_equ[0] - cells->speed0[mappings[0]]);
        av_local_density += temp;
        result->speed0[mappings[0]] = temp;

        temp = cells->speed1[mappings[1]] + params.omega * (d_equ[1] - cells->speed1[mappings[1]]);
        av_local_density += temp;
        result->speed1[mappings[1]] = temp;

        temp = cells->speed2[mappings[2]] + params.omega * (d_equ[2] - cells->speed2[mappings[2]]);
        av_local_density += temp;
        result->speed2[mappings[2]] = temp;

        temp = cells->speed3[mappings[3]] + params.omega * (d_equ[3] - cells->speed3[mappings[3]]);
        av_local_density += temp;
        result->speed3[mappings[3]] = temp;

        temp = cells->speed4[mappings[4]] + params.omega * (d_equ[4] - cells->speed4[mappings[4]]);
        av_local_density += temp;
        result->speed4[mappings[4]] = temp;

        temp = cells->speed5[mappings[5]] + params.omega * (d_equ[5] - cells->speed5[mappings[5]]);
        av_local_density += temp;
        result->speed5[mappings[5]] = temp;

        temp = cells->speed6[mappings[6]] + params.omega * (d_equ[6] - cells->speed6[mappings[6]]);
        av_local_density += temp;
        result->speed6[mappings[6]] = temp;

        temp = cells->speed7[mappings[7]] + params.omega * (d_equ[7] - cells->speed7[mappings[7]]);
        av_local_density += temp;
        result->speed7[mappings[7]] = temp;

        temp = cells->speed8[mappings[8]] + params.omega * (d_equ[8] - cells->speed8[mappings[8]]);
        av_local_density += temp;
        result->speed8[mappings[8]] = temp;

        /* x-component of velocity */
        float av_u_x = (result->speed1[ii + jj*params.nx]
                      + result->speed5[ii + jj*params.nx]
                      + result->speed8[ii + jj*params.nx]
                      - (result->speed3[ii + jj*params.nx]
                         + result->speed6[ii + jj*params.nx]
                         + result->speed7[ii + jj*params.nx]))
                     / av_local_density;
        /* compute y velocity component */
        float av_u_y = (result->speed2[ii + jj*params.nx]
                      + result->speed5[ii + jj*params.nx]
                      + result->speed6[ii + jj*params.nx]
                      - (result->speed4[ii + jj*params.nx]
                         + result->speed7[ii + jj*params.nx]
                         + result->speed8[ii + jj*params.nx]))
                     / av_local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((av_u_x * av_u_x) + (av_u_y * av_u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
      else{
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        result->speed1[ii + jj*params.nx] = cells->speed3[mappings[3]];
        result->speed2[ii + jj*params.nx] = cells->speed4[mappings[4]];
        result->speed3[ii + jj*params.nx] = cells->speed1[mappings[1]];
        result->speed4[ii + jj*params.nx] = cells->speed2[mappings[2]];
        result->speed5[ii + jj*params.nx] = cells->speed7[mappings[7]];
        result->speed6[ii + jj*params.nx] = cells->speed8[mappings[8]];
        result->speed7[ii + jj*params.nx] = cells->speed5[mappings[5]];
        result->speed8[ii + jj*params.nx] = cells->speed6[mappings[6]];
      }
    }
  }

  return tot_u / (float)tot_cells;
}

// float av_velocity(const t_param params, speeds* cells, int* obstacles)
// {
//   int tot_cells = 0;  /* no. of cells used in calculation */
//   float tot_u;          /* accumulated magnitudes of velocity for each cell */

//   /* initialise */
//   tot_u = 0.f;

//   /* loop over all non-blocked cells */
//   for (int jj = 0; jj < params.ny; jj++)
//   {
//     for (int ii = 0; ii < params.nx; ii++)
//     {
//       /* ignore occupied cells */
//       if (!obstacles[ii + jj*params.nx])
//       {
//         /* local density total */
//         float local_density = 0.f;

//         for (int kk = 0; kk < NSPEEDS; kk++)
//         {
//           local_density += cells[ii + jj*params.nx].speeds[kk];
//         }

//         /* x-component of velocity */
//         float u_x = (cells[ii + jj*params.nx].speeds[1]
//                       + cells[ii + jj*params.nx].speeds[5]
//                       + cells[ii + jj*params.nx].speeds[8]
//                       - (cells[ii + jj*params.nx].speeds[3]
//                          + cells[ii + jj*params.nx].speeds[6]
//                          + cells[ii + jj*params.nx].speeds[7]))
//                      / local_density;
//         /* compute y velocity component */
//         float u_y = (cells[ii + jj*params.nx].speeds[2]
//                       + cells[ii + jj*params.nx].speeds[5]
//                       + cells[ii + jj*params.nx].speeds[6]
//                       - (cells[ii + jj*params.nx].speeds[4]
//                          + cells[ii + jj*params.nx].speeds[7]
//                          + cells[ii + jj*params.nx].speeds[8]))
//                      / local_density;
//         /* accumulate the norm of x- and y- velocity components */
//         tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
//         /* increase counter of inspected cells */
//         ++tot_cells;
//       }
//     }
//   }

//   return tot_u / (float)tot_cells;
// }

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, speeds* cells_ptr, speeds* tmp_cells_ptr,
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
  cells_ptr->speed0 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed1 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed2 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed3 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed4 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed5 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed6 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed7 = malloc(sizeof(float) * params->nx * params->ny);
  cells_ptr->speed8 = malloc(sizeof(float) * params->nx * params->ny);

  if (cells_ptr->speed0 == NULL || cells_ptr->speed8 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speed0 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed1 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed2 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed3 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed4 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed5 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed6 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed7 = malloc(sizeof(float) * params->nx * params->ny);
  tmp_cells_ptr->speed8 = malloc(sizeof(float) * params->nx * params->ny);

  if (tmp_cells_ptr->speed0 == NULL || tmp_cells_ptr->speed8 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  cells_ptr->speed0[0] = w0;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {


      /* centre */
      cells_ptr->speed0[ii + jj*params->nx] = w0;
      /* axis directions */
      cells_ptr->speed1[ii + jj*params->nx] = w1;
      cells_ptr->speed2[ii + jj*params->nx] = w1;
      cells_ptr->speed3[ii + jj*params->nx] = w1;
      cells_ptr->speed4[ii + jj*params->nx] = w1;
      /* diagonals */
      cells_ptr->speed5[ii + jj*params->nx] = w2;
      cells_ptr->speed6[ii + jj*params->nx] = w2;
      cells_ptr->speed7[ii + jj*params->nx] = w2;
      cells_ptr->speed8[ii + jj*params->nx] = w2;
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
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, speeds* cells_ptr, speeds* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(cells_ptr->speed0);
  cells_ptr->speed0 = NULL;

  free(cells_ptr->speed1);
  cells_ptr->speed1 = NULL;

  free(cells_ptr->speed2);
  cells_ptr->speed2 = NULL;

  free(cells_ptr->speed3);
  cells_ptr->speed3 = NULL;

  free(cells_ptr->speed4);
  cells_ptr->speed4 = NULL;

  free(cells_ptr->speed5);
  cells_ptr->speed5 = NULL;

  free(cells_ptr->speed6);
  cells_ptr->speed6 = NULL;

  free(cells_ptr->speed7);
  cells_ptr->speed7 = NULL;

  free(cells_ptr->speed8);
  cells_ptr->speed8 = NULL;

  free(tmp_cells_ptr->speed0);
  tmp_cells_ptr->speed0 = NULL;

  free(tmp_cells_ptr->speed1);
  tmp_cells_ptr->speed1 = NULL;

  free(tmp_cells_ptr->speed2);
  tmp_cells_ptr->speed2 = NULL;

  free(tmp_cells_ptr->speed3);
  tmp_cells_ptr->speed3 = NULL;

  free(tmp_cells_ptr->speed4);
  tmp_cells_ptr->speed4 = NULL;

  free(tmp_cells_ptr->speed5);
  tmp_cells_ptr->speed5 = NULL;

  free(tmp_cells_ptr->speed6);
  tmp_cells_ptr->speed6 = NULL;

  free(tmp_cells_ptr->speed7);
  tmp_cells_ptr->speed7 = NULL;

  free(tmp_cells_ptr->speed8);
  tmp_cells_ptr->speed8 = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free(cells_ptr);
  cells_ptr = NULL;

  free(tmp_cells_ptr);
  tmp_cells_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float average)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return average * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, speeds* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total = cells->speed0[ii + jj*params.nx]
              + cells->speed1[ii + jj*params.nx]
              + cells->speed2[ii + jj*params.nx]
              + cells->speed3[ii + jj*params.nx]
              + cells->speed4[ii + jj*params.nx]
              + cells->speed5[ii + jj*params.nx]
              + cells->speed6[ii + jj*params.nx] 
              + cells->speed7[ii + jj*params.nx]
              + cells->speed8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, speeds* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = cells->speed0[ii + jj*params.nx] 
                      + cells->speed1[ii + jj*params.nx]
                      + cells->speed2[ii + jj*params.nx]
                      + cells->speed3[ii + jj*params.nx]
                      + cells->speed4[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed6[ii + jj*params.nx]
                      + cells->speed7[ii + jj*params.nx]
                      + cells->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        u_x = (cells->speed1[ii + jj*params.nx]
                + cells->speed5[ii + jj*params.nx]
                + cells->speed8[ii + jj*params.nx]
                - (cells->speed3[ii + jj*params.nx]
                + cells->speed6[ii + jj*params.nx]
                + cells->speed7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speed2[ii + jj*params.nx]
                + cells->speed5[ii + jj*params.nx]
                + cells->speed6[ii + jj*params.nx]
                - (cells->speed4[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]
                  + cells->speed8[ii + jj*params.nx]))
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