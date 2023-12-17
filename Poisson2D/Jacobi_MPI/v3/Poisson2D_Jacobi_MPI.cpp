#include <stdio.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <mpi.h>

#define PI M_PI
#define idx(i, j) ((i + 1) * (subdomain.ny + 2) + (j + 1))

// Poisson's equation (-(u_xx+u_yy) = RHS) in 2D with Dirichlet boundary conditions
// Discretized with finite differences
// Jacobi Iteration
// parallelized with domain decomposition, MPI

double exact_solution_func(double x, double y)
{
    return x * (x - 2) * sin(y);
}

double RHS_func(double x, double y) // negative Laplacian of the exact solution
{
    return -2 * sin(y) + x * (x - 2) * sin(y);
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

void exchangeBoundaryData(const MPIEnv &env, const Subdomain &subdomain, std::vector<double> &local_sol, MPI_Datatype column_data_type)
{
    if (env.back_neighbor != MPI_PROC_NULL)
    {
        MPI_Sendrecv(&local_sol[idx(0, 0)], subdomain.ny, MPI_DOUBLE, env.back_neighbor, 0,
                     &local_sol[idx(-1, 0)], subdomain.ny, MPI_DOUBLE, env.back_neighbor, 0,
                     env.cart_comm, MPI_STATUS_IGNORE);
    }
    if (env.front_neighbor != MPI_PROC_NULL)
    {
        MPI_Sendrecv(&local_sol[idx(subdomain.nx - 1, 0)], subdomain.ny, MPI_DOUBLE, env.front_neighbor, 0,
                     &local_sol[idx(subdomain.nx, 0)], subdomain.ny, MPI_DOUBLE, env.front_neighbor, 0,
                     env.cart_comm, MPI_STATUS_IGNORE);
    }
    if (env.left_neighbor != MPI_PROC_NULL)
    {
        MPI_Sendrecv(&local_sol[idx(0, 0)], 1, column_data_type, env.left_neighbor, 0,
                     &local_sol[idx(0, -1)], 1, column_data_type, env.left_neighbor, 0,
                     env.cart_comm, MPI_STATUS_IGNORE);
    }
    if (env.right_neighbor != MPI_PROC_NULL)
    {
        MPI_Sendrecv(&local_sol[idx(0, subdomain.ny - 1)], 1, column_data_type, env.right_neighbor, 0,
                     &local_sol[idx(0, subdomain.ny)], 1, column_data_type, env.right_neighbor, 0,
                     env.cart_comm, MPI_STATUS_IGNORE);
    }
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
    const double a1 = -1.0, b1 = 1.0, a2 = -1.5, b2 = 2.5;   // [a1, b1] x [a2, b2]
    const size_t Nx = 367, Ny = 491;                         // Grid size in x and y direction
    const size_t numPoints_x = Nx + 1, numPoints_y = Ny + 1; // Number of points in x and y direction
    const double tolerance = 1e-15;                          // Tolerance for convergence
    const unsigned int maxIter = 100000000;                  // Maximum number of iterations
    const double dx = (b1 - a1) / Nx, dy = (b2 - a2) / Ny;

    // Initialize MPI and create Cartesian topology for the processes
    MPIEnv env;
    initializeMPI(argc, argv, env);

    // Initialize and setup subdomain information
    Subdomain subdomain;
    setupSubdomain(env, subdomain, numPoints_x, numPoints_y);

    // Store local vectors in contiguous memory, including ghost points
    std::vector<double> sol((subdomain.nx + 2) * (subdomain.ny + 2)), sol_old((subdomain.nx + 2) * (subdomain.ny + 2)),
        exact_sol((subdomain.nx + 2) * (subdomain.ny + 2)), RHS((subdomain.nx + 2) * (subdomain.ny + 2));

    // Initialize exact solution and RHS
    initializeProblemData(a1, b1, a2, b2, dx, dy, subdomain, exact_sol, RHS);

    // Set initial guess for solution
    setupInitialGuess(subdomain, sol);

    // Apply Dirichlet boundary conditions
    applyDirichletBC(env, subdomain, sol, exact_sol);

    // Initialize the iteration
    sol_old = sol;
    unsigned int iter = 0;
    double global_maxAbsDiff;
    // Preparations for exchanging boundary data
    double dx2_recipr = 1.0 / (dx * dx), dy2_recipr = 1.0 / (dy * dy), denom = 2.0 * (dx2_recipr + dy2_recipr);
    MPI_Datatype column_data_type; // MPI derived data type for column data
    MPI_Type_vector(subdomain.nx, 1, subdomain.ny + 2, MPI_DOUBLE, &column_data_type);
    MPI_Type_commit(&column_data_type);

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();
    do
    {
        if (env.rank == 0 && (iter == 1 || (iter % 1000 == 0 && iter != 0)))
        {
            printf("Iteration %8u: max absolute difference = %e\n", iter, global_maxAbsDiff);
        }

        double maxAbsDiff = 0.0;

        // Send and receive boundary data
        exchangeBoundaryData(env, subdomain, sol_old, column_data_type);

        // Jacobi iteration for 2D
        // Update interior points in each rank
        for (int i = 1; i < subdomain.nx - 1; ++i)
        {
            for (int j = 1; j < subdomain.ny - 1; ++j)
            {
                sol[idx(i, j)] = (dx2_recipr * (sol_old[idx(i - 1, j)] + sol_old[idx(i + 1, j)]) +
                                  dy2_recipr * (sol_old[idx(i, j - 1)] + sol_old[idx(i, j + 1)]) +
                                  RHS[idx(i, j)]) /
                                 denom;
                maxAbsDiff = std::max(maxAbsDiff, std::abs(sol[idx(i, j)] - sol_old[idx(i, j)]));
            }
        }

        // Update boundary points in each rank
        // Back boundary (if not at domain back boundary)
        if (env.back_neighbor != MPI_PROC_NULL)
        {
            int i = 0;
            for (int j = 0; j < subdomain.ny; ++j)
            {
                if ((j == 0 && env.left_neighbor != MPI_PROC_NULL) || (j == subdomain.ny - 1 && env.right_neighbor != MPI_PROC_NULL) || (j > 0 && j < subdomain.ny - 1))
                {
                    sol[idx(i, j)] = (dx2_recipr * (sol_old[idx(i - 1, j)] + sol_old[idx(i + 1, j)]) +
                                      dy2_recipr * (sol_old[idx(i, j - 1)] + sol_old[idx(i, j + 1)]) +
                                      RHS[idx(i, j)]) /
                                     denom;
                    maxAbsDiff = std::max(maxAbsDiff, std::abs(sol[idx(i, j)] - sol_old[idx(i, j)]));
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
                    sol[idx(i, j)] = (dx2_recipr * (sol_old[idx(i - 1, j)] + sol_old[idx(i + 1, j)]) +
                                      dy2_recipr * (sol_old[idx(i, j - 1)] + sol_old[idx(i, j + 1)]) +
                                      RHS[idx(i, j)]) /
                                     denom;
                    maxAbsDiff = std::max(maxAbsDiff, std::abs(sol[idx(i, j)] - sol_old[idx(i, j)]));
                }
            }
        }
        // Left boundary (if not at domain left boundary)
        if (env.left_neighbor != MPI_PROC_NULL)
        {
            int j = 0;
            for (int i = 1; i < subdomain.nx - 1; ++i)
            {
                sol[idx(i, j)] = (dx2_recipr * (sol_old[idx(i - 1, j)] + sol_old[idx(i + 1, j)]) +
                                  dy2_recipr * (sol_old[idx(i, j - 1)] + sol_old[idx(i, j + 1)]) +
                                  RHS[idx(i, j)]) /
                                 denom;
                maxAbsDiff = std::max(maxAbsDiff, std::abs(sol[idx(i, j)] - sol_old[idx(i, j)]));
            }
        }
        // Right boundary (if not at domain right boundary)
        if (env.right_neighbor != MPI_PROC_NULL)
        {
            int j = subdomain.ny - 1;
            for (int i = 1; i < subdomain.nx - 1; ++i)
            {
                sol[idx(i, j)] = (dx2_recipr * (sol_old[idx(i - 1, j)] + sol_old[idx(i + 1, j)]) +
                                  dy2_recipr * (sol_old[idx(i, j - 1)] + sol_old[idx(i, j + 1)]) +
                                  RHS[idx(i, j)]) /
                                 denom;
                maxAbsDiff = std::max(maxAbsDiff, std::abs(sol[idx(i, j)] - sol_old[idx(i, j)]));
            }
        }
        MPI_Allreduce(&maxAbsDiff, &global_maxAbsDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        std::swap(sol, sol_old);
        iter++;
    } while (iter < maxIter && global_maxAbsDiff > tolerance);
    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Calculate error
    double global_maxError = calculateError(subdomain, sol, exact_sol);

    if (env.rank == 0)
    {
        printf("Converged after %u iterations with max error %e\n", iter, global_maxError);
        printf("Elapsed time: %lf seconds\n", elapsed_time.count());
    }

    MPI_Type_free(&column_data_type);
    MPI_Finalize();

    return 0;
}
