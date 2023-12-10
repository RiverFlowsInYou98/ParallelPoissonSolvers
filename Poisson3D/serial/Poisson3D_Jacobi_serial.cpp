#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define PI M_PI

// Poisson's equation (-(u_xx+u_yy+u_zz) = RHS) in 3D with Dirichlet boundary conditions
// Discretized with finite differences
// Jacobi Iteration
// serial implementation

double exact_solution_func(double x, double y, double z)
{
    return x * (x - 2) * sin(y) * cos(z);
    // return sin(PI * x) * sin(PI * y) * sin(PI * z);
}

double RHS_func(double x, double y, double z) // negative Laplacian of exact solution
{
    return 2 * (x * (x - 2) - 1) * sin(y) * cos(z);
    // return 3 * PI * PI * sin(PI * x) * sin(PI * y) * sin(PI * z);
}

int main()
{
    const double a1 = -1.0, b1 = 1.0, a2 = -1.5, b2 = 1.2, a3 = -2.0, b3 = 1.5;    // [a1, b1] x [a2, b2] x [a3, b3]
    const size_t Nx = 87, Ny = 141, Nz = 189;                                      // Grid size in x, y and z direction
    const size_t numPoints_x = Nx + 1, numPoints_y = Ny + 1, numPoints_z = Nz + 1; // Number of points in x, y and z direction
    const double tolerance = 1e-14;                                                // Tolerance for convergence
    const unsigned int maxIter = 100000000;                                        // Maximum number of iterations
    const double dx = (b1 - a1) / Nx, dy = (b2 - a2) / Ny, dz = (b3 - a3) / Nz;    // Grid spacing in x, y and z direction

    size_t totalPoints = numPoints_x * numPoints_y * numPoints_z;
    std::vector<double> sol(totalPoints, 0.0), sol_old(totalPoints, 0.0), exact_solution(totalPoints, 0.0), RHS(totalPoints, 0.0);

    auto index = [&](size_t i, size_t j, size_t k) -> size_t
    {
        return i * numPoints_y * numPoints_z + j * numPoints_z + k;
    };

    // Initialize exact solution and RHS
    for (size_t i = 0; i < numPoints_x; ++i)
    {
        for (size_t j = 0; j < numPoints_y; ++j)
        {
            for (size_t k = 0; k < numPoints_z; ++k)
            {
                double x = a1 + i * dx, y = a2 + j * dy, z = a3 + k * dz;
                exact_solution[index(i, j, k)] = exact_solution_func(x, y, z);
                RHS[index(i, j, k)] = RHS_func(x, y, z);
            }
        }
    }

    // Set initial guess for solution
    // std::fill(sol.begin(), sol.end(), 0.0);
    for (size_t i = 0; i < numPoints_x; ++i)
    {
        for (size_t j = 0; j < numPoints_y; ++j)
        {
            for (size_t k = 0; k < numPoints_z; ++k)
            {
                sol[index(i, j, k)] = 0.0;
            }
        }
    }

    // Apply Dirichlet boundary conditions
    for (size_t i = 0; i < numPoints_x; ++i)
    {
        for (size_t j = 0; j < numPoints_y; ++j)
        {
            for (size_t k = 0; k < numPoints_z; ++k)
            {
                if (i == 0 || i == Nx || j == 0 || j == Ny || k == 0 || k == Nz)
                {
                    sol[index(i, j, k)] = exact_solution[index(i, j, k)];
                }
            }
        }
    }

    // Store previous iteration
    sol_old = sol;

    unsigned int iter = 0;
    double maxAbsDiff = 0.0;
    double dx2_recipr = 1.0 / (dx * dx), dy2_recipr = 1.0 / (dy * dy), dz2_recipr = 1.0 / (dz * dz);
    double denom = 2.0 * (dx2_recipr + dy2_recipr + dz2_recipr);

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        if (iter == 1 || (iter % 1000 == 0 && iter != 0))
        {
            printf("Iteration %8u: max absolute difference = %e\n", iter, maxAbsDiff);
        }
        maxAbsDiff = 0.0;
        // Jacobi iteration for 3D
        for (size_t i = 1; i < Nx; ++i)
        {
            for (size_t j = 1; j < Ny; ++j)
            {
                for (size_t k = 1; k < Nz; ++k)
                {
                    sol[index(i, j, k)] = (dy2_recipr * (sol_old[index(i - 1, j, k)] + sol_old[index(i + 1, j, k)]) +
                                           dx2_recipr * (sol_old[index(i, j - 1, k)] + sol_old[index(i, j + 1, k)]) +
                                           dz2_recipr * (sol_old[index(i, j, k - 1)] + sol_old[index(i, j, k + 1)]) +
                                           RHS[index(i, j, k)]) /
                                          denom;
                    maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[index(i, j, k)] - sol_old[index(i, j, k)]));
                }
            }
        }
        // Swap sol and sol_old vectors
        std::swap(sol, sol_old);
        iter++;
    } while (iter < maxIter && maxAbsDiff > tolerance);

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Compute the maximum error
    double maxError = 0.0;
    for (size_t i = 0; i < numPoints_x; i++)
    {
        for (size_t j = 0; j < numPoints_y; j++)
        {
            for (size_t k = 0; k < numPoints_z; k++)
            {
                maxError = std::max(maxError, std::fabs(sol_old[index(i, j, k)] - exact_solution[index(i, j, k)]));
            }
        }
    }

    printf("Converged after %u iterations with max error %e\n", iter, maxError);
    printf("Elapsed time: %.6f seconds\n", elapsed_time.count());
    return 0;
}
