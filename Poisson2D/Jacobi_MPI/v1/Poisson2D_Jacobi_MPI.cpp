#include <stdio.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <mpi.h>

#define PI M_PI
#define idx(i, j) ((i) * local_ny + (j))

// Poisson's equation (-(u_xx+u_yy) = RHS) in 2D with Dirichlet boundary conditions
// Discretized with finite differences
// Jacobi Iteration
// parallelized with domain decomposition, MPI

double exact_solution_func(double x, double y)
{
    return x * (x - 2) * sin(y);
}

double RHS_func(double x, double y) // negative Laplacian of exact solution
{
    return -2 * sin(y) + x * (x - 2) * sin(y);
}

int main(int argc, char **argv)
{
    const double a1 = -1.0, b1 = 1.0, a2 = -1.5, b2 = 2.5;   // [a1, b1] x [a2, b2]
    const size_t Nx = 367, Ny = 491;                         // Grid size in x and y direction
    const size_t numPoints_x = Nx + 1, numPoints_y = Ny + 1; // Number of points in x and y direction
    const double tolerance = 1e-15;                          // Tolerance for convergence
    const unsigned int maxIter = 100000000;                  // Maximum number of iterations
    const double dx = (b1 - a1) / Nx, dy = (b2 - a2) / Ny;

    // Initialize MPI
    int comm_rank = -1, comm_size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int dims[2] = {0, 0};
    MPI_Dims_create(comm_size, 2, dims);
    int num_procs_x = dims[0], num_procs_y = dims[1]; // Number of processes in x, y-direction
    // if (comm_rank == 0)
    // {
    //     printf("Number of processes: %d\n", comm_size);
    //     printf("Number of processes in x-direction: %d\n", num_procs_x);
    //     printf("Number of processes in y-direction: %d\n", num_procs_y);
    // }
    // Create Cartesian topology for the processes
    MPI_Comm cart_comm;
    int periods[2] = {0, 0}; // Non-periodic boundary conditions
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int top_neighbor, bottom_neighbor, left_neighbor, right_neighbor;
    MPI_Cart_shift(cart_comm, 0, 1, &top_neighbor, &bottom_neighbor);
    MPI_Cart_shift(cart_comm, 1, 1, &left_neighbor, &right_neighbor);
    // printf("Process %d has top neighbor %d and bottom neighbor %d\n", comm_rank, top_neighbor, bottom_neighbor);
    // printf("Process %d has left neighbor %d and right neighbor %d\n", comm_rank, left_neighbor, right_neighbor);
    // Get the coordinates of the current process in the Cartesian topology
    int coords[2];
    MPI_Cart_coords(cart_comm, comm_rank, 2, coords);
    int x_rank = coords[0], y_rank = coords[1]; // x, y-coordinate in the process grid
    int remainder_x = numPoints_x % num_procs_x, remainder_y = numPoints_y % num_procs_y;
    int local_nx = numPoints_x / num_procs_x + (x_rank < remainder_x ? 1 : 0);
    int local_ny = numPoints_y / num_procs_y + (y_rank < remainder_y ? 1 : 0);
    int local_size = local_nx * local_ny;
    int local_start_x = x_rank * (numPoints_x / num_procs_x) + std::min(x_rank, remainder_x);
    int local_start_y = y_rank * (numPoints_y / num_procs_y) + std::min(y_rank, remainder_y);
    int local_end_x = local_start_x + local_nx;
    int local_end_y = local_start_y + local_ny;

    // // Print the rank's message
    // printf("Process %d of %d has coordinates (%d, %d) and local size (%d, %d), x range [%d, %d], y range [%d, %d]\n",
    //        comm_rank, comm_size, x_rank, y_rank, local_nx, local_ny, local_start_x, local_end_x - 1, local_start_y, local_end_y - 1);

    // Store local vectors in contiguous memory
    std::vector<double> local_sol(local_size), local_sol_old(local_size),
        local_exact_solution(local_size), local_RHS(local_size);

    // Initialize exact solution and RHS
    for (int i = 0; i < local_nx; i++)
    {
        for (int j = 0; j < local_ny; j++)
        {
            int global_index_x = local_start_x + i;
            int global_index_y = local_start_y + j;
            double x = a1 + global_index_x * dx, y = a2 + global_index_y * dy;
            local_exact_solution[idx(i, j)] = exact_solution_func(x, y);
            local_RHS[idx(i, j)] = RHS_func(x, y);
        }
    }

    // Set initial guess for solution
    for (int i = 0; i < local_nx; ++i)
    {
        for (int j = 0; j < local_ny; ++j)
        {
            local_sol[idx(i, j)] = 0.0;
        }
    }

    // Apply Dirichlet boundary conditions
    if (top_neighbor == MPI_PROC_NULL) // x_rank == 0
    {
        for (int j = 0; j < local_ny; ++j)
        {
            local_sol[idx(0, j)] = local_exact_solution[idx(0, j)];
        }
    }
    if (bottom_neighbor == MPI_PROC_NULL) // x_rank == num_procs_x - 1
    {
        for (int j = 0; j < local_ny; ++j)
        {
            local_sol[idx(local_nx - 1, j)] = local_exact_solution[idx(local_nx - 1, j)];
        }
    }
    if (left_neighbor == MPI_PROC_NULL) // y_rank == 0
    {
        for (int i = 0; i < local_nx; ++i)
        {
            local_sol[idx(i, 0)] = local_exact_solution[idx(i, 0)];
        }
    }
    if (right_neighbor == MPI_PROC_NULL) // y_rank == num_procs_y - 1
    {
        for (int i = 0; i < local_nx; ++i)
        {
            local_sol[idx(i, local_ny - 1)] = local_exact_solution[idx(i, local_ny - 1)];
        }
    }
    // Initialize the iteration
    local_sol_old = local_sol;
    unsigned int iter = 0;
    double maxAbsDiff;

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();
    do
    {
        if (comm_rank == 0 && (iter == 1 || (iter % 1000 == 0 && iter != 0)))
        {
            printf("Iteration %8u: max absolute difference = %e\n", iter, maxAbsDiff);
        }

        double local_maxAbsDiff = 0.0;

        // Send and receive boundary data
        std::vector<double> received_top_boundary_data(local_ny), received_bottom_boundary_data(local_ny), received_left_boundary_data(local_nx), received_right_boundary_data(local_nx);
        if (top_neighbor != MPI_PROC_NULL)
        {
            MPI_Sendrecv(&local_sol_old[idx(0, 0)], local_ny, MPI_DOUBLE, top_neighbor, 0,
                         &received_top_boundary_data[0], local_ny, MPI_DOUBLE, top_neighbor, 0,
                         cart_comm, MPI_STATUS_IGNORE);
        }
        if (bottom_neighbor != MPI_PROC_NULL)
        {
            MPI_Sendrecv(&local_sol_old[idx(local_nx - 1, 0)], local_ny, MPI_DOUBLE, bottom_neighbor, 0,
                         &received_bottom_boundary_data[0], local_ny, MPI_DOUBLE, bottom_neighbor, 0,
                         cart_comm, MPI_STATUS_IGNORE);
        }
        if (left_neighbor != MPI_PROC_NULL)
        {
            std::vector<double> sent_left_boundary_data(local_nx);
            for (int i = 0; i < local_nx; ++i)
            {
                sent_left_boundary_data[i] = local_sol_old[idx(i, 0)];
            }
            MPI_Sendrecv(sent_left_boundary_data.data(), local_nx, MPI_DOUBLE, left_neighbor, 0,
                         &received_left_boundary_data[0], local_nx, MPI_DOUBLE, left_neighbor, 0,
                         cart_comm, MPI_STATUS_IGNORE);
        }
        if (right_neighbor != MPI_PROC_NULL)
        {
            std::vector<double> sent_right_boundary_data(local_nx);
            for (int i = 0; i < local_nx; ++i)
            {
                sent_right_boundary_data[i] = local_sol_old[idx(i, local_ny - 1)];
            }
            MPI_Sendrecv(sent_right_boundary_data.data(), local_nx, MPI_DOUBLE, right_neighbor, 0,
                         &received_right_boundary_data[0], local_nx, MPI_DOUBLE, right_neighbor, 0,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // Jacobi iteration for 2D
        // Update interior points in each rank
        for (int i = 1; i < local_nx - 1; ++i)
        {
            for (int j = 1; j < local_ny - 1; ++j)
            {
                local_sol[idx(i, j)] = (dy * dy * (local_sol_old[idx(i - 1, j)] + local_sol_old[idx(i + 1, j)]) +
                                   dx * dx * (local_sol_old[idx(i, j - 1)] + local_sol_old[idx(i, j + 1)]) +
                                   dx * dx * dy * dy * local_RHS[idx(i, j)]) /
                                  (2 * (dx * dx + dy * dy));
                local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(i, j)] - local_sol_old[idx(i, j)]));
            }
        }

        // Update boundary points in each rank
        // Top boundary (if not at domain top boundary)
        if (top_neighbor != MPI_PROC_NULL)
        {
            for (int j = 0; j < local_ny; ++j)
            {
                if (j == 0 && left_neighbor != MPI_PROC_NULL)
                {
                    local_sol[idx(0, 0)] = (dy * dy * (received_top_boundary_data[0] + local_sol_old[idx(1, 0)]) +
                                       dx * dx * (received_left_boundary_data[0] + local_sol_old[idx(0, 1)]) +
                                       dx * dx * dy * dy * local_RHS[idx(0, 0)]) /
                                      (2 * (dx * dx + dy * dy));
                    local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(0, 0)] - local_sol_old[idx(0, 0)]));
                }
                else if (j == local_ny - 1 && right_neighbor != MPI_PROC_NULL)
                {
                    local_sol[idx(0, local_ny - 1)] = (dy * dy * (received_top_boundary_data[local_ny - 1] + local_sol_old[idx(1, local_ny - 1)]) +
                                                  dx * dx * (local_sol_old[idx(0, local_ny - 2)] + received_right_boundary_data[0]) +
                                                  dx * dx * dy * dy * local_RHS[idx(0, local_ny - 1)]) /
                                                 (2 * (dx * dx + dy * dy));
                    local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(0, local_ny - 1)] - local_sol_old[idx(0, local_ny - 1)]));
                }
                else if (j > 0 && j < local_ny - 1)
                {
                    local_sol[idx(0, j)] = (dy * dy * (received_top_boundary_data[j] + local_sol_old[idx(1, j)]) +
                                       dx * dx * (local_sol_old[idx(0, j - 1)] + local_sol_old[idx(0, j + 1)]) +
                                       dx * dx * dy * dy * local_RHS[idx(0, j)]) /
                                      (2 * (dx * dx + dy * dy));
                    local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(0, j)] - local_sol_old[idx(0, j)]));
                }
            }
        }
        // Bottom boundary (if not at domain bottom boundary)
        if (bottom_neighbor != MPI_PROC_NULL)
        {
            for (int j = 0; j < local_ny; ++j)
            {
                if (j == 0 && left_neighbor != MPI_PROC_NULL)
                {
                    local_sol[idx(local_nx - 1, 0)] = (dy * dy * (local_sol_old[idx(local_nx - 2, 0)] + received_bottom_boundary_data[0]) +
                                                  dx * dx * (received_left_boundary_data[local_nx - 1] + local_sol_old[idx(local_nx - 1, 1)]) +
                                                  dx * dx * dy * dy * local_RHS[idx(local_nx - 1, 0)]) /
                                                 (2 * (dx * dx + dy * dy));
                    local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(local_nx - 1, 0)] - local_sol_old[idx(local_nx - 1, 0)]));
                }
                else if (j == local_ny - 1 && right_neighbor != MPI_PROC_NULL)
                {
                    local_sol[idx(local_nx - 1, local_ny - 1)] = (dy * dy * (local_sol_old[idx(local_nx - 2, local_ny - 1)] + received_bottom_boundary_data[local_ny - 1]) +
                                                             dx * dx * (local_sol_old[idx(local_nx - 1, local_ny - 2)] + received_right_boundary_data[local_nx - 1]) +
                                                             dx * dx * dy * dy * local_RHS[idx(local_nx - 1, local_ny - 1)]) /
                                                            (2 * (dx * dx + dy * dy));
                    local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(local_nx - 1, local_ny - 1)] - local_sol_old[idx(local_nx - 1, local_ny - 1)]));
                }
                else if (j > 0 && j < local_ny - 1)
                {
                    local_sol[idx(local_nx - 1, j)] = (dy * dy * (local_sol_old[idx(local_nx - 2, j)] + received_bottom_boundary_data[j]) +
                                                  dx * dx * (local_sol_old[idx(local_nx - 1, j - 1)] + local_sol_old[idx(local_nx - 1, j + 1)]) +
                                                  dx * dx * dy * dy * local_RHS[idx(local_nx - 1, j)]) /
                                                 (2 * (dx * dx + dy * dy));
                    local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(local_nx - 1, j)] - local_sol_old[idx(local_nx - 1, j)]));
                }
            }
        }
        if (left_neighbor != MPI_PROC_NULL)
        {
            for (int i = 1; i < local_nx - 1; ++i)
            {

                local_sol[idx(i, 0)] = (dy * dy * (local_sol_old[idx(i - 1, 0)] + local_sol_old[idx(i + 1, 0)]) +
                                   dx * dx * (received_left_boundary_data[i] + local_sol_old[idx(i, 1)]) +
                                   dx * dx * dy * dy * local_RHS[idx(i, 0)]) /
                                  (2 * (dx * dx + dy * dy));
                local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(i, 0)] - local_sol_old[idx(i, 0)]));
            }
        }
        // Right boundary (if not at domain right boundary)
        if (right_neighbor != MPI_PROC_NULL)
        {
            for (int i = 1; i < local_nx - 1; ++i)
            {
                local_sol[idx(i, local_ny - 1)] = (dy * dy * (local_sol_old[idx(i - 1, local_ny - 1)] + local_sol_old[idx(i + 1, local_ny - 1)]) +
                                              dx * dx * (local_sol_old[idx(i, local_ny - 2)] + received_right_boundary_data[i]) +
                                              dx * dx * dy * dy * local_RHS[idx(i, local_ny - 1)]) /
                                             (2 * (dx * dx + dy * dy));
                local_maxAbsDiff = std::max(local_maxAbsDiff, std::abs(local_sol[idx(i, local_ny - 1)] - local_sol_old[idx(i, local_ny - 1)]));
            }
        }
        MPI_Allreduce(&local_maxAbsDiff, &maxAbsDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        std::swap(local_sol, local_sol_old);
        iter++;
    } while (iter < maxIter && maxAbsDiff > tolerance);
    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Calculate local maximum error
    double local_maxError = 0.0;
    for (int i = 0; i < local_nx; i++)
    {
        for (int j = 0; j < local_ny; j++)
        {
            local_maxError = std::max(local_maxError, std::fabs(local_sol[idx(i, j)] - local_exact_solution[idx(i, j)]));
        }
    }
    // Use MPI_Reduce to find the maximum local error
    double global_maxError;
    MPI_Reduce(&local_maxError, &global_maxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (comm_rank == 0)
    {
        printf("Converged after %u iterations with max error %e\n", iter, global_maxError);
        printf("Elapsed time: %lf seconds\n", elapsed_time.count());
    }

    MPI_Finalize();

    return 0;
}
