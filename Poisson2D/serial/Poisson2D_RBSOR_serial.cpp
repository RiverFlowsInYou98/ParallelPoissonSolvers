#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define PI M_PI

// Poisson's equation (-(u_xx+u_yy) = RHS) in 2D with Dirichlet boundary conditions
// Discretized with finite differences
// Red-Black Successive Over-Relaxation (SOR) Iteration
// serial implementation

double exact_solution_func(double x, double y)
{
    return x * (x - 2) * sin(y);
}

double RHS_func(double x, double y) // negative Laplacian of the exact solution
{
    return -2 * sin(y) + x * (x - 2) * sin(y);
}

int main()
{
    const double a1 = -1.0, b1 = 1.0, a2 = -1.5, b2 = 2.5;   // [a1, b1] x [a2, b2]
    const size_t Nx = 367, Ny = 491;                         // Grid size in x and y direction
    const size_t numPoints_x = Nx + 1, numPoints_y = Ny + 1; // Number of points in x and y direction
    const double tolerance = 1e-15;                          // Tolerance for convergence
    const unsigned int maxIter = 100000000;                  // Maximum number of iterations
    const double dx = (b1 - a1) / Nx, dy = (b2 - a2) / Ny;
    const double omega = 1.5;                                // Relaxation factor for SOR

    std::vector<std::vector<double>> sol(numPoints_x, std::vector<double>(numPoints_y)),
        exact_solution(numPoints_x, std::vector<double>(numPoints_y)),
        RHS(numPoints_x, std::vector<double>(numPoints_y));

    // Initialize exact solution and RHS
    for (size_t i = 0; i < numPoints_x; ++i)
    {
        for (size_t j = 0; j < numPoints_y; ++j)
        {
            double x = a1 + i * dx, y = a2 + j * dy;
            exact_solution[i][j] = exact_solution_func(x, y);
            RHS[i][j] = RHS_func(x, y);
        }
    }

    // Set initial guess for solution
    for (size_t i = 0; i < numPoints_x; ++i)
    {
        for (size_t j = 0; j < numPoints_y; ++j)
        {
            sol[i][j] = 0.0;
        }
    }

    // Apply Dirichlet boundary conditions
    for (size_t i = 0; i < numPoints_x; ++i)
    {
        sol[i][0] = exact_solution[i][0];
        sol[i][Ny] = exact_solution[i][Ny];
    }
    for (size_t j = 0; j < numPoints_y; ++j)
    {
        sol[0][j] = exact_solution[0][j];
        sol[Nx][j] = exact_solution[Nx][j];
    }

    unsigned int iter = 0;
    double maxAbsDiff = 0.0;

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        if (iter == 1 || (iter % 1000 == 0 && iter != 0))
        {
            printf("Iteration %8u: max absolute difference = %e\n", iter, maxAbsDiff);
        }
        maxAbsDiff = 0.0;

        // Red-Black SOR iteration for 2D
        for (int color = 0; color <= 1; color++)
        {
            for (size_t i = 1; i < Nx; ++i)
            {
                for (size_t j = 1; j < Ny; ++j)
                {
                    if ((i + j) % 2 == color)
                    { // Checkerboard pattern
                        double gs_update = (dy * dy * (sol[i - 1][j] + sol[i + 1][j]) +
                                            dx * dx * (sol[i][j - 1] + sol[i][j + 1]) +
                                            dx * dx * dy * dy * RHS[i][j]) /
                                           (2 * (dx * dx + dy * dy));
                        double new_val = sol[i][j] + omega * (gs_update - sol[i][j]);
                        maxAbsDiff = std::max(maxAbsDiff, std::fabs(new_val - sol[i][j]));
                        sol[i][j] = new_val;
                    }
                }
            }
        }
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
            maxError = std::max(maxError, std::fabs(sol[i][j] - exact_solution[i][j]));
        }
    }

    printf("Converged after %u iterations with max error %e\n", iter, maxError);
    printf("Elapsed time: %.6f seconds\n", elapsed_time.count());

    return 0;
}
