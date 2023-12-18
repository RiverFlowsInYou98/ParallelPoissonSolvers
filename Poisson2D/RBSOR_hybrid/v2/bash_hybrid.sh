#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=1:00:00

## For my account MaxCpuPerUserLimit=32
#SBATCH -N 1
#SBATCH --constraint=32core|intel|cascade|edr
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-core=1

#SBATCH -J MyParallelJob

# Specify an output file
#SBATCH -o Hybrid-%j.out
#SBATCH -e Hybrid-%j.out

# Load necessary modules
module load gcc/10.2 cuda/11.7.1
module load mpi/openmpi_4.1.1_gcc_10.2_slurm22


lscpu

# Set the number of threads for OpenMP (adjust as needed)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Number of MPI tasks: " $SLURM_NTASKS
echo "Number of OpenMP threads per MPI task: " $OMP_NUM_THREADS

# # Compile without OpenMP
# mpic++ -O3 Poisson2D_RBSOR_hybrid.cpp -o Poisson2D_RBSOR_hybrid.out
# Compile with OpenMP
mpic++ -O3 -fopenmp Poisson2D_RBSOR_hybrid.cpp -o Poisson2D_RBSOR_hybrid.out

echo -e "Compile Done\n"

# Run the application, need to set Nx and Ny
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 10 10
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 20 20
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 40 40
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 80 80
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 160 160
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 320 320
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 640 640
echo
srun --mpi=pmix ./Poisson2D_RBSOR_hybrid.out 1280 1280


