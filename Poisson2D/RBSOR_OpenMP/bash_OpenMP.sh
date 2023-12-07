#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=48:00:00

#SBATCH -N 1
#SBATCH --constraint=48core|intel|cascade|edr
#SBATCH -n 1
#SBATCH --cpus-per-task=2

# Specify a job name:
#SBATCH -J OpenMP

# Specify an output file
#SBATCH -o OpenMP-%j.out
#SBATCH -e OpenMP-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Solve Poisson equation in 2D with finite difference and Red-Black SOR iteration, OpenMP parallelization"
g++ -O3 -fopenmp Poisson2D_RBSOR_OpenMP.cpp -o Poisson2D_RBSOR_OpenMP.out
echo "Compile done"
./Poisson2D_RBSOR_OpenMP.out