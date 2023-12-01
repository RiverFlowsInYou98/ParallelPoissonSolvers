#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=1:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1

# Specify a job name:
#SBATCH -J MyMPIJob

# Specify an output file
#SBATCH -o MPI-%j.out
#SBATCH -e MPI-%j.out


module load gcc/10.2 cuda/11.7.1
module load mpi/openmpi_4.1.1_gcc_10.2_slurm22

# Run a command
# srun --mpi=list
# srun --cpu-bind=help

mpic++ -O3 Poisson1D_Jacobi_MPI.cpp -o Poisson1D_Jacobi_MPI.out
echo "Compiled"
srun --mpi=pmix ./Poisson1D_Jacobi_MPI.out
