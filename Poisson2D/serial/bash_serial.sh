#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=48:00:00

#SBATCH -N 1
#SBATCH --constraint=32core|intel|cascade|edr
#SBATCH -n 1

# Specify a job name:
#SBATCH -J serial

# Specify an output file
#SBATCH -o serial-%j.out
#SBATCH -e serial-%j.out


# echo "Solve Poisson equation in 2D with finite difference and Jacobi iteration, serial version"
# g++ -O3 Poisson2D_Jacobi_serial.cpp -o Poisson2D_Jacobi_serial.out
# ./Poisson2D_Jacobi_serial.out

# echo "Solve Poisson equation in 2D with finite difference and SOR iteration, serial version"
# g++ -O3 Poisson2D_SOR_serial.cpp -o Poisson2D_SOR_serial.out
# ./Poisson2D_SOR_serial.out

echo "Solve Poisson equation in 2D with finite difference and Red-Black SOR iteration, serial version"
g++ -O3 Poisson2D_RBSOR_serial.cpp -o Poisson2D_RBSOR_serial.out
echo "Compile done"
./Poisson2D_RBSOR_serial.out