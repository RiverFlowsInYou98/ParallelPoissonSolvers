#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define PI M_PI

// Poisson's equation (-u_xx = RHS) in 1D with Dirichlet boundary conditions
// Discretized with finite differences
// Jacobi Iteration
// serial implementation

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

int main()
{
    const double a = -1.0, b = 1.0;         // Interval [a,b] in x direction
    const size_t N = 1000;                    // Grid size in x direction
    const size_t numPoints = N + 1;         // Number of points in x direction
    const double tolerance = 1e-15;         // Tolerance for convergence
    const unsigned int maxIter = 100000000; // Maximum number of iterations
    const double dx = (b - a) / N;
    std::vector<double> sol(numPoints), sol_old(numPoints), exact_solution(numPoints), RHS(numPoints);

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

    // Store previous iteration
    sol_old = sol;

    unsigned int iter = 0;
    double maxAbsDiff;

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        // if (iter < 10)
        // {
        //     printf("Iteration %d: ", iter);
        //     for (size_t i = 0; i < numPoints; ++i)
        //     {
        //         printf("%f ", sol_old[i]);
        //     }
        //     printf("\n");
        // }
        maxAbsDiff = 0.0;
        for (size_t i = 1; i < N; ++i)
        {
            // Jacobi iteration
            sol[i] = 0.5 * (dx * dx * RHS[i] + sol_old[i - 1] + sol_old[i + 1]);
            maxAbsDiff = std::max(maxAbsDiff, std::fabs(sol[i] - sol_old[i]));
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
    for (size_t i = 0; i < numPoints; i++)
    {
        maxError = std::max(maxError, std::fabs(sol_old[i] - exact_solution[i]));
    }

    std::cout << "Converged after " << iter << " iterations with max error " << maxError << std::endl;
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
