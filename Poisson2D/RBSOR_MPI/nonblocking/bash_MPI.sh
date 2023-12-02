#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=1:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH -N 2
#SBATCH --tasks-per-node=8
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

mpic++ -O3 Poisson2D_RBSOR_MPI.cpp -o Poisson2D_RBSOR_MPI.out
echo "Compiled"
srun --mpi=pmix ./Poisson2D_RBSOR_MPI.out
