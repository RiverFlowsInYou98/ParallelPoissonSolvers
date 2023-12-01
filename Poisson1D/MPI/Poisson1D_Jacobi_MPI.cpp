#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <mpi.h>

#define PI M_PI

// Poisson's equation (-u_xx = RHS) in 1D with Dirichlet boundary conditions
// Discretized with finite differences
// Jacobi Iteration
// parallelized with domain decomposition, MPI

double exact_solution_func(double x)
{
    // return cos(2.0 * PI * x) * exp(-2 * x);
    return x * sin(2.0 * PI * x);
}

double RHS_func(double x)
{
    // return -exp(-2.0 * x) * (8 * PI * sin(2 * PI * x) - 4 * (PI * PI - 1) * cos(2 * PI * x));
    return 4 * PI * PI * x * sin(2.0 * PI * x) - 4 * PI * cos(2.0 * PI * x);
}

int main(int argc, char **argv)
{
    const double a = -1.0, b = 1.0;         // Interval [a,b] in x direction
    const size_t N = 1000;                    // Grid size in x direction
    const size_t numPoints = N + 1;         // Total number of points
    const double tolerance = 1e-10;         // Tolerance for convergence
    const unsigned int maxIter = 100000000; // Maximum number of iterations
    const double dx = (b - a) / N;

    // Initialize MPI
    int comm_rank = -1, comm_size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int remainder = numPoints % comm_size;                                          // Remainder points
    int local_chunk_size = numPoints / comm_size + (comm_rank < remainder ? 1 : 0); // Number of points per process
    // Calculate local_start and local_end
    int local_start = comm_rank * (numPoints / comm_size) + std::min(comm_rank, remainder);
    int local_end = local_start + local_chunk_size;

    // Use std::vector to store local data
    std::vector<double> local_x(local_chunk_size), local_RHS(local_chunk_size), local_exact_solution(local_chunk_size), local_sol(local_chunk_size), local_sol_old(local_chunk_size);

    // Populate local x and f values
    for (int i = 0; i < local_chunk_size; i++)
    {
        int global_index = local_start + i;
        local_x[i] = a + global_index * dx;
        local_RHS[i] = RHS_func(local_x[i]);
        local_exact_solution[i] = exact_solution_func(local_x[i]);
    }

    // Set initial guess for solution
    for (int i = 0; i < local_chunk_size; ++i)
    {
        local_sol[i] = 0.0;
    }

    // boundary condition
    if (comm_rank == 0)
    {
        local_sol[0] = local_exact_solution[0];
    }
    else if (comm_rank == comm_size - 1)
    {
        local_sol[local_chunk_size - 1] = local_exact_solution[local_chunk_size - 1];
    }

    // previous iteration
    local_sol_old = local_sol;

    unsigned int iter = 0;
    double maxAbsDiff;

    // // Define the displacements and receive counts for Gatherv
    // std::vector<int> displs(comm_size, 0);
    // std::vector<int> recvcounts(comm_size, 0);
    // int total_size = 0;
    // for (int i = 0; i < comm_size; ++i)
    // {
    //     recvcounts[i] = numPoints / comm_size + (i < remainder ? 1 : 0);
    //     displs[i] = total_size;
    //     total_size += recvcounts[i];
    // }
    // std::vector<double> gathered_sol_old(total_size);

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        // if (comm_rank == 0 && iter % 100 == 0)
        // {
        //     std::cout << "Iteration " << iter << " with max error " << maxAbsDiff << std::endl;
        // }
        // if (iter < 10)
        // {
        //     // Gather the sol_old vectors from all processes
        //     MPI_Gatherv(local_sol_old.data(), local_chunk_size, MPI_DOUBLE, gathered_sol_old.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //     // Root process prints the gathered vector
        //     if (comm_rank == 0)
        //     {
        //         printf("Iteration %d: ", iter);
        //         for (int i = 0; i < displs[comm_size - 1] + recvcounts[comm_size - 1]; ++i)
        //         {
        //             printf("%f ", gathered_sol_old[i]);
        //         }
        //         printf("\n");
        //     }
        // }

        maxAbsDiff = 0.0;
        double local_maxAbsDiff = 0.0;
        double left_boundary_data, right_boundary_data;

        if (comm_rank > 0)
        {
            // Send the a[0] of comm_rank to prev_rank as prev_rank's a[chunk_size]
            MPI_Send(&local_sol_old[0], 1, MPI_DOUBLE, comm_rank - 1, 0, MPI_COMM_WORLD);
            // Receive the a[chunk_size-1] of prev_rank as comm_rank's a[-1]
            MPI_Recv(&left_boundary_data, 1, MPI_DOUBLE, comm_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (comm_rank < comm_size - 1)
        {
            // Send the a[chunk_size-1] of comm_rank to next_rank as next_rank's a[-1]
            MPI_Send(&local_sol_old[local_chunk_size - 1], 1, MPI_DOUBLE, comm_rank + 1, 0, MPI_COMM_WORLD);
            // Receive the a[0] of next_rank as comm_rank's a[chunk_size]
            MPI_Recv(&right_boundary_data, 1, MPI_DOUBLE, comm_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        local_sol[0] = (comm_rank == 0) ? local_exact_solution[0] : 0.5 * (dx * dx * local_RHS[0] + left_boundary_data + local_sol_old[1]);
        local_maxAbsDiff = std::max(local_maxAbsDiff, std::fabs(local_sol[0] - local_sol_old[0]));
        for (int i = 1; i < local_chunk_size - 1; ++i)
        {
            // Use Jacobi iteration
            local_sol[i] = 0.5 * (dx * dx * local_RHS[i] + local_sol_old[i - 1] + local_sol_old[i + 1]);
            local_maxAbsDiff = std::max(local_maxAbsDiff, std::fabs(local_sol[i] - local_sol_old[i]));
        }

        local_sol[local_chunk_size - 1] = (comm_rank == comm_size - 1) ? local_exact_solution[local_chunk_size - 1] : 0.5 * (dx * dx * local_RHS[local_chunk_size - 1] + local_sol_old[local_chunk_size - 2] + right_boundary_data);
        local_maxAbsDiff = std::max(local_maxAbsDiff, std::fabs(local_sol[local_chunk_size - 1] - local_sol_old[local_chunk_size - 1]));

        MPI_Allreduce(&local_maxAbsDiff, &maxAbsDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        // Swap sol1 and sol2 vectors
        std::swap(local_sol, local_sol_old);
        iter++;
    } while (iter < maxIter && maxAbsDiff > tolerance);
    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    // Calculate local maximum error
    double local_maxError = 0.0;
    for (int i = 0; i < local_chunk_size; i++)
    {
        local_maxError = std::max(local_maxError, std::fabs(local_sol[i] - local_exact_solution[i]));
    }
    // Use MPI_Reduce to find the maximum local error
    double global_maxError;
    MPI_Reduce(&local_maxError, &global_maxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (comm_rank == 0)
    {
        std::cout << "Converged after " << iter << " iterations with max error " << global_maxError << std::endl;
        std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}