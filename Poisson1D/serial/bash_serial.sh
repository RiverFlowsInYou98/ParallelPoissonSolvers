echo "Solve Poisson equation in 1D with finite difference and Jacobi iteration, serial version"
g++ -O3 Poisson1D_Jacobi_serial.cpp -o Poisson1D_Jacobi_serial.out
./Poisson1D_Jacobi_serial.out

echo "Solve Poisson equation in 1D with finite difference and Gauss-Seidel iteration, serial version"
g++ -O3 Poisson1D_GS_serial.cpp -o Poisson1D_GS_serial.out
./Poisson1D_GS_serial.out

echo "Solve Poisson equation in 1D with finite difference and Red-Black Gauss-Seidel iteration, serial version"
g++ -O3 Poisson1D_RBGS_serial.cpp -o Poisson1D_RBGS_serial.out
./Poisson1D_RBGS_serial.out

echo "Solve Poisson equation in 1D with finite difference and Red-Black SOR iteration, serial version"
g++ -O3 Poisson1D_RBSOR_serial.cpp -o Poisson1D_RBSOR_serial.out
./Poisson1D_RBSOR_serial.out