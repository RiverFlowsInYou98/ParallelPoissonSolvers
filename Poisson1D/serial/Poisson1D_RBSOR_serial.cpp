#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define PI M_PI

// Poisson's equation (-u_xx = RHS) in 1D with Dirichlet boundary conditions
// Discretized with finite differences
// Red-Black Successive Over-Relaxation (SOR) Iteration
// serial implementation

double exact_solution_func(double x)
{
    return x * sin(2.0 * PI * x);
}

double RHS_func(double x)
{
    return 4 * PI * PI * x * sin(2.0 * PI * x) - 4 * PI * cos(2.0 * PI * x);
}

int main()
{
    const double a = -1.0, b = 1.0;         // Interval [a,b] in x direction
    const size_t N = 1000;                  // Grid size in x direction
    const size_t numPoints = N + 1;         // Number of points in x direction
    const double tolerance = 1e-10;         // Tolerance for convergence
    const unsigned int maxIter = 100000000; // Maximum number of iterations
    const double dx = (b - a) / N;
    const double omega = 1.5; // Relaxation factor for SOR
    std::vector<double> sol(numPoints), exact_solution(numPoints), RHS(numPoints);

    // Initialize exact solution and RHS
    for (size_t i = 0; i < numPoints; ++i)
    {
        double x = a + i * dx;
        exact_solution[i] = exact_solution_func(x);
        RHS[i] = RHS_func(x);
    }

    // Set initial guess for solution
    std::fill(sol.begin(), sol.end(), 0.0);

    // Apply Dirichlet boundary conditions
    sol.front() = exact_solution.front();
    sol.back() = exact_solution.back();

    unsigned int iter = 0;
    double maxAbsDiff;

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        maxAbsDiff = 0.0;

        // Update Red Points
        for (size_t i = 1; i < N; i += 2)
        { // Assuming red points are at odd indices
            double new_val = sol[i] + omega * (0.5 * (dx * dx * RHS[i] + sol[i - 1] + sol[i + 1]) - sol[i]);
            maxAbsDiff = std::max(maxAbsDiff, std::fabs(new_val - sol[i]));
            sol[i] = new_val;
        }

        // Update Black Points
        for (size_t i = 2; i < N - 1; i += 2)
        { // Assuming black points are at even indices
            double new_val = sol[i] + omega * (0.5 * (dx * dx * RHS[i] + sol[i - 1] + sol[i + 1]) - sol[i]);
            maxAbsDiff = std::max(maxAbsDiff, std::fabs(new_val - sol[i]));
            sol[i] = new_val;
        }

        iter++;
    } while (iter < maxIter && maxAbsDiff > tolerance);

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Compute the maximum error
    double maxError = 0.0;
    for (size_t i = 0; i < numPoints; i++)
    {
        maxError = std::max(maxError, std::fabs(sol[i] - exact_solution[i]));
    }

    std::cout << "Converged after " << iter << " iterations with max error " << maxError << std::endl;
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
