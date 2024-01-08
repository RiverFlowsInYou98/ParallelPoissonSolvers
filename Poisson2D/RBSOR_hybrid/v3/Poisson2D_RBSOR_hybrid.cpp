#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#define PI M_PI
#define idx(i, j) ((i + 1) * (subdomain.ny + 2) + (j + 1))

// Poisson's equation (-(u_xx+u_yy) = RHS) in 2D with Dirichlet boundary conditions
// Discretized with finite differences
// Red-Black Successive Over-Relaxation (SOR) Iteration
// Hybrid parallelization with OpenMP and domain decomposition, MPI

double exact_solution_func(double x, double y)
{
    // return x * (x - 2) * sin(y);
    return sin(x) * sin(y);
}

double RHS_func(double x, double y) // negative Laplacian of the exact solution
{
    // return -2 * sin(y) + x * (x - 2) * sin(y);
    return 2 * sin(x) * sin(y);
}

struct MPIEnv
{
    int size, rank;
    MPI_Comm cart_comm;
    int num_procs_x, num_procs_y, rank_x, rank_y;
    int back_neighbor, front_neighbor, left_neighbor, right_neighbor;
    MPIEnv() : size(0), rank(-1), cart_comm(MPI_COMM_NULL),
               num_procs_x(0), num_procs_y(0), rank_x(-1), rank_y(-1),
               back_neighbor(MPI_PROC_NULL), front_neighbor(MPI_PROC_NULL),
               left_neighbor(MPI_PROC_NULL), right_neighbor(MPI_PROC_NULL) {}
};

struct Subdomain
{
    int nx, ny;
    int start_x, end_x, start_y, end_y;
    Subdomain() : nx(0), ny(0), start_x(0), end_x(0), start_y(0), end_y(0) {}
};

void initializeMPI(int argc, char **argv, MPIEnv &env)
{
    /*
    Example of 2D Cartesian topology: 16=4x4 processes
    x increases from back to front
    y increases from left to right
    rank (rank_x, rank_y), rank = num_procs_y * rank_x + rank_y
    -------------------------------------------------------------------> y
    |  rank0  (0,0) |  rank1  (0,1) |  rank2  (0,2) |  rank3  (0,3) |
    -----------------------------------------------------------------
    |  rank4  (1,0) |  rank5  (1,1) |  rank6  (1,2) |  rank7  (1,3) |
    -----------------------------------------------------------------
    |  rank8  (2,0) |  rank9  (2,1) |  rank10 (2,2) |  rank11 (2,3) |
    -----------------------------------------------------------------
    |  rank12 (3,0) |  rank13 (3,1) |  rank14 (3,2) |  rank15 (3,3) |
    -----------------------------------------------------------------
    |
    v x
    */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &env.size);

    int dims[2] = {0, 0};
    MPI_Dims_create(env.size, 2, dims);
    env.num_procs_x = dims[0];
    env.num_procs_y = dims[1];

    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &env.cart_comm);

    MPI_Cart_shift(env.cart_comm, 0, 1, &env.back_neighbor, &env.front_neighbor);
    MPI_Cart_shift(env.cart_comm, 1, 1, &env.left_neighbor, &env.right_neighbor);

    // x, y-coordinate in the process grid
    int coords[2];
    MPI_Cart_coords(env.cart_comm, env.rank, 2, coords);
    env.rank_x = coords[0];
    env.rank_y = coords[1];
}

void setupSubdomain(const MPIEnv &env, Subdomain &subdomain, const size_t numPoints_x, const size_t numPoints_y)
{
    int remainder_x = numPoints_x % env.num_procs_x, remainder_y = numPoints_y % env.num_procs_y;
    subdomain.nx = numPoints_x / env.num_procs_x + (env.rank_x < remainder_x ? 1 : 0);
    subdomain.ny = numPoints_y / env.num_procs_y + (env.rank_y < remainder_y ? 1 : 0);

    subdomain.start_x = env.rank_x * (numPoints_x / env.num_procs_x) + std::min(env.rank_x, remainder_x);
    subdomain.start_y = env.rank_y * (numPoints_y / env.num_procs_y) + std::min(env.rank_y, remainder_y);

    subdomain.end_x = subdomain.start_x + subdomain.nx;
    subdomain.end_y = subdomain.start_y + subdomain.ny;
}

void initializeProblemData(const double a1, const double b1, const double a2, const double b2, const double dx, const double dy,
                           const Subdomain &subdomain, std::vector<double> &exact_sol, std::vector<double> &RHS)
{
    for (int i = 0; i < subdomain.nx; i++)
    {
        for (int j = 0; j < subdomain.ny; j++)
        {
            int global_index_x = subdomain.start_x + i;
            int global_index_y = subdomain.start_y + j;
            double x = a1 + global_index_x * dx, y = a2 + global_index_y * dy;
            exact_sol[idx(i, j)] = exact_solution_func(x, y);
            RHS[idx(i, j)] = RHS_func(x, y);
        }
    }
}

void setupInitialGuess(const Subdomain &subdomain, std::vector<double> &sol)
{
    for (int i = 0; i < subdomain.nx; ++i)
    {
        for (int j = 0; j < subdomain.ny; ++j)
        {
            sol[idx(i, j)] = 0.0;
        }
    }
}

void applyDirichletBC(const MPIEnv &env, const Subdomain &subdomain, std::vector<double> &sol, const std::vector<double> &exact_sol)
{
    if (env.back_neighbor == MPI_PROC_NULL) // rank_x == 0
    {
        for (int j = 0; j < subdomain.ny; ++j)
        {
            sol[idx(0, j)] = exact_sol[idx(0, j)];
        }
    }
    if (env.front_neighbor == MPI_PROC_NULL) // rank_x == num_procs_x - 1
    {
        for (int j = 0; j < subdomain.ny; ++j)
        {
            sol[idx(subdomain.nx - 1, j)] = exact_sol[idx(subdomain.nx - 1, j)];
        }
    }
    if (env.left_neighbor == MPI_PROC_NULL) // rank_y == 0
    {
        for (int i = 0; i < subdomain.nx; ++i)
        {
            sol[idx(i, 0)] = exact_sol[idx(i, 0)];
        }
    }
    if (env.right_neighbor == MPI_PROC_NULL) // rank_y == num_procs_y - 1
    {
        for (int i = 0; i < subdomain.nx; ++i)
        {
            sol[idx(i, subdomain.ny - 1)] = exact_sol[idx(i, subdomain.ny - 1)];
        }
    }
}

void startExchangeBoundaryDataNonblocking(const MPIEnv &env, const Subdomain &subdomain, std::vector<double> &local_sol,
                                          MPI_Datatype column_data_type, std::vector<MPI_Request> &requests)
{
    int request_count = 0;
    // Start non-blocking receive
    if (env.back_neighbor != MPI_PROC_NULL)
        MPI_Irecv(&local_sol[idx(-1, 0)], subdomain.ny, MPI_DOUBLE, env.back_neighbor, 0, env.cart_comm, &requests[request_count++]);
    if (env.front_neighbor != MPI_PROC_NULL)
        MPI_Irecv(&local_sol[idx(subdomain.nx, 0)], subdomain.ny, MPI_DOUBLE, env.front_neighbor, 0, env.cart_comm, &requests[request_count++]);
    if (env.left_neighbor != MPI_PROC_NULL)
        MPI_Irecv(&local_sol[idx(0, -1)], 1, column_data_type, env.left_neighbor, 0, env.cart_comm, &requests[request_count++]);
    if (env.right_neighbor != MPI_PROC_NULL)
        MPI_Irecv(&local_sol[idx(0, subdomain.ny)], 1, column_data_type, env.right_neighbor, 0, env.cart_comm, &requests[request_count++]);
    // Start non-blocking send
    if (env.back_neighbor != MPI_PROC_NULL)
        MPI_Isend(&local_sol[idx(0, 0)], subdomain.ny, MPI_DOUBLE, env.back_neighbor, 0, env.cart_comm, &requests[request_count++]);
    if (env.front_neighbor != MPI_PROC_NULL)
        MPI_Isend(&local_sol[idx(subdomain.nx - 1, 0)], subdomain.ny, MPI_DOUBLE, env.front_neighbor, 0, env.cart_comm, &requests[request_count++]);
    if (env.left_neighbor != MPI_PROC_NULL)
        MPI_Isend(&local_sol[idx(0, 0)], 1, column_data_type, env.left_neighbor, 0, env.cart_comm, &requests[request_count++]);
    if (env.right_neighbor != MPI_PROC_NULL)
        MPI_Isend(&local_sol[idx(0, subdomain.ny - 1)], 1, column_data_type, env.right_neighbor, 0, env.cart_comm, &requests[request_count++]);
}

void completeExchangeBoundaryDataNonblocking(std::vector<MPI_Request> &requests)
{
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
}

double calculateError(const Subdomain &subdomain, const std::vector<double> &sol, const std::vector<double> &exact_sol)
{
    // Calculate local error
    double local_maxError = 0.0;
    for (int i = 0; i < subdomain.nx; i++)
    {
        for (int j = 0; j < subdomain.ny; j++)
        {
            local_maxError = std::max(local_maxError, std::fabs(sol[idx(i, j)] - exact_sol[idx(i, j)]));
        }
    }
    // Use MPI_Reduce to find the maximum local error
    double global_maxError;
    MPI_Reduce(&local_maxError, &global_maxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global_maxError;
}

int main(int argc, char **argv)
{
    // Get Nx and Ny, Grid size in x and y direction
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " Nx Ny" << std::endl;
        return 1;
    }

    int Nx = std::atoi(argv[1]);
    int Ny = std::atoi(argv[2]);

    // Ensure Nx and Ny are valid
    if (Nx <= 0 || Ny <= 0)
    {
        std::cerr << "Nx and Ny must be positive integers" << std::endl;
        return 1;
    }

    const double a1 = -1.0, b1 = 1.0, a2 = -1.0, b2 = 1.0;   // [a1, b1] x [a2, b2]
    const size_t numPoints_x = Nx + 1, numPoints_y = Ny + 1; // Number of points in x and y direction
    const double tolerance = 1e-14;                          // Tolerance for convergence
    const unsigned int maxIter = 100000000;                  // Maximum number of iterations
    const double dx = (b1 - a1) / Nx, dy = (b2 - a2) / Ny;   // Grid spacing in x and y direction
    const double omega = 1.5;                                // Relaxation factor for SOR
    // // Optimal relaxation factor for this problem
    // const double rho = (cos(PI / Nx) + (dx * dx / dy * dy) * cos(PI / Ny)) / (1 + (dx * dx / dy * dy));
    // const double omega = 2.0 / (1.0 + sqrt(1.0 - rho * rho));

    // Initialize MPI and create Cartesian topology for the processes
    MPIEnv env;
    initializeMPI(argc, argv, env);

    // Initialize and setup subdomain information
    Subdomain subdomain;
    setupSubdomain(env, subdomain, numPoints_x, numPoints_y);
    // // Print the rank's message
    // printf("\nMPI rank %d:\n", env.rank);
    // if (env.rank == 0)
    // {
    //     printf("Number of processes: %d\n", env.size);
    //     printf("Number of processes in x and y direction: %d, %d\n", env.num_procs_x, env.num_procs_y);
    // }
    // printf("Process %d of %d has coordinates (%d, %d) and local size %d x %d = %d, x range [%d, %d], y range [%d, %d]\n",
    //        env.rank, env.size, env.rank_x, env.rank_y, subdomain.nx, subdomain.ny, subdomain.local_numPoints,
    //        subdomain.start_x, subdomain.end_x - 1, subdomain.start_y, subdomain.end_y - 1);
    // printf("Process %d has back neighbor %d, front neighbor %d, left neighbor %d and right neighbor %d\n",
    //        env.rank, env.back_neighbor, env.front_neighbor, env.left_neighbor, env.right_neighbor);

    // Store local vectors in contiguous memory, including ghost points
    std::vector<double> sol((subdomain.nx + 2) * (subdomain.ny + 2)),
        exact_sol((subdomain.nx + 2) * (subdomain.ny + 2)), RHS((subdomain.nx + 2) * (subdomain.ny + 2));

    // Initialize exact solution and RHS
    initializeProblemData(a1, b1, a2, b2, dx, dy, subdomain, exact_sol, RHS);

    // Set initial guess for solution
    setupInitialGuess(subdomain, sol);

    // Apply Dirichlet boundary conditions
    applyDirichletBC(env, subdomain, sol, exact_sol);

    // Initialize the iteration
    unsigned int iter = 0;
    double maxAbsDiff, global_maxAbsDiff;
    double gs_update, sor_update;
    // Preparations for exchanging boundary data
    std::vector<MPI_Request> requests(8, MPI_REQUEST_NULL);
    const double dx2_recipr = 1.0 / (dx * dx), dy2_recipr = 1.0 / (dy * dy), denom = 2.0 * (dx2_recipr + dy2_recipr);
    const double w1 = dx2_recipr / denom, w2 = dy2_recipr / denom, w3 = 1 / denom;
    MPI_Datatype column_data_type; // MPI derived data type for column data
    MPI_Type_vector(subdomain.nx, 1, subdomain.ny + 2, MPI_DOUBLE, &column_data_type);
    MPI_Type_commit(&column_data_type);

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        // if (env.rank == 0 && (iter == 1 || (iter % 1000 == 0 && iter != 0)))
        // {
        //     printf("Iteration %8u: max absolute difference = %e\n", iter, global_maxAbsDiff);
        // }

        maxAbsDiff = 0.0;
        // Red-Black SOR iteration for 2D
        for (int color = 0; color <= 1; color++)
        {
            startExchangeBoundaryDataNonblocking(env, subdomain, sol, column_data_type, requests);

            // Update interior points in each rank
            #pragma omp parallel for reduction(max : maxAbsDiff) private(gs_update, sor_update)
            for (int i = 1; i < subdomain.nx - 1; ++i)
            {
                for (int j = 1; j < subdomain.ny - 1; ++j)
                {
                    if ((subdomain.start_x + i + subdomain.start_y + j) % 2 == color)
                    {
                        gs_update = w1 * (sol[idx(i - 1, j)] + sol[idx(i + 1, j)]) +
                                    w2 * (sol[idx(i, j - 1)] + sol[idx(i, j + 1)]) +
                                    w3 * RHS[idx(i, j)];
                        sor_update = sol[idx(i, j)] + omega * (gs_update - sol[idx(i, j)]);
                        maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[idx(i, j)] - sor_update));
                        sol[idx(i, j)] = sor_update;
                    }
                }
            }

            // Complete boundary data exchange
            completeExchangeBoundaryDataNonblocking(requests);

            // Update boundary points in each rank
            // Back boundary (if not at domain back boundary)
            if (env.back_neighbor != MPI_PROC_NULL)
            {
                int i = 0;
                for (int j = 0; j < subdomain.ny; ++j)
                {
                    if ((j == 0 && env.left_neighbor != MPI_PROC_NULL) || (j == subdomain.ny - 1 && env.right_neighbor != MPI_PROC_NULL) || (j > 0 && j < subdomain.ny - 1))
                    {
                        if ((subdomain.start_x + i + subdomain.start_y + j) % 2 == color)
                        {
                            gs_update = w1 * (sol[idx(i - 1, j)] + sol[idx(i + 1, j)]) +
                                        w2 * (sol[idx(i, j - 1)] + sol[idx(i, j + 1)]) +
                                        w3 * RHS[idx(i, j)];
                            sor_update = sol[idx(i, j)] + omega * (gs_update - sol[idx(i, j)]);
                            maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[idx(i, j)] - sor_update));
                            sol[idx(i, j)] = sor_update;
                        }
                    }
                }
            }
            // Front boundary (if not at domain front boundary)
            if (env.front_neighbor != MPI_PROC_NULL)
            {
                int i = subdomain.nx - 1;
                for (int j = 0; j < subdomain.ny; ++j)
                {
                    if ((j == 0 && env.left_neighbor != MPI_PROC_NULL) || (j == subdomain.ny - 1 && env.right_neighbor != MPI_PROC_NULL) || (j > 0 && j < subdomain.ny - 1))
                    {
                        if ((subdomain.start_x + i + subdomain.start_y + j) % 2 == color)
                        {
                            gs_update = w1 * (sol[idx(i - 1, j)] + sol[idx(i + 1, j)]) +
                                        w2 * (sol[idx(i, j - 1)] + sol[idx(i, j + 1)]) +
                                        w3 * RHS[idx(i, j)];
                            sor_update = sol[idx(i, j)] + omega * (gs_update - sol[idx(i, j)]);
                            maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[idx(i, j)] - sor_update));
                            sol[idx(i, j)] = sor_update;
                        }
                    }
                }
            }
            // Left boundary (if not at domain left boundary)
            if (env.left_neighbor != MPI_PROC_NULL)
            {
                int j = 0;
                for (int i = 1; i < subdomain.nx - 1; ++i)
                {
                    if ((subdomain.start_x + i + subdomain.start_y + j) % 2 == color)
                    {
                        gs_update = w1 * (sol[idx(i - 1, j)] + sol[idx(i + 1, j)]) +
                                    w2 * (sol[idx(i, j - 1)] + sol[idx(i, j + 1)]) +
                                    w3 * RHS[idx(i, j)];
                        sor_update = sol[idx(i, j)] + omega * (gs_update - sol[idx(i, j)]);
                        maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[idx(i, j)] - sor_update));
                        sol[idx(i, j)] = sor_update;
                    }
                }
            }
            // Right boundary (if not at domain right boundary)
            if (env.right_neighbor != MPI_PROC_NULL)
            {
                int j = subdomain.ny - 1;
                for (int i = 1; i < subdomain.nx - 1; ++i)
                {
                    if ((subdomain.start_x + i + subdomain.start_y + j) % 2 == color)
                    {
                        gs_update = w1 * (sol[idx(i - 1, j)] + sol[idx(i + 1, j)]) +
                                    w2 * (sol[idx(i, j - 1)] + sol[idx(i, j + 1)]) +
                                    w3 * RHS[idx(i, j)];
                        sor_update = sol[idx(i, j)] + omega * (gs_update - sol[idx(i, j)]);
                        maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[idx(i, j)] - sor_update));
                        sol[idx(i, j)] = sor_update;
                    }
                }
            }
        }
        MPI_Allreduce(&maxAbsDiff, &global_maxAbsDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iter++;
    } while (iter < maxIter && global_maxAbsDiff > tolerance);
    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Calculate error
    double global_maxError = calculateError(subdomain, sol, exact_sol);

    if (env.rank == 0)
    {
        printf("Nx = %d, Ny = %d\n", Nx, Ny);
        printf("Converged after %u iterations with max error %e\n", iter, global_maxError);
        printf("Elapsed time: %lf seconds\n", elapsed_time.count());
    }

    MPI_Type_free(&column_data_type);
    MPI_Finalize();

    return 0;
}
