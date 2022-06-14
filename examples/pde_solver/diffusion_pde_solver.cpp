
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <iomanip>
#include <model.h>
#include <surrogate.h>
#include "feedforward_model.h"

template <typename T>
void zero_matrix(T* arr, int nrows, int ncols) {
    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            arr[idx] = 0;
        }
    }
}

constexpr double PI = 3.1415926589793238;

void make_forcing_term(double* f, int n) {
    // Forcing function assumes an n*n grid with (h,h) in top left corner and (n*h,n*h) in bottom right
    // Once we include the boundary conditions this gives us a square with corners at (0,0) to (1,1)
    double h = 1.0/(n+1);
    for (int r=0; r<n; ++r) {
        for (int c=0; c<n; ++c) {
            double x = (r+1) * h;
            double y = (c+1) * h;
            f[r*n+c] = -2*PI*PI*sin(PI*x)*sin(PI*y);
        }
    }
}

void make_boundary(double* T, int n, double value) {
    for (int i=0; i<n+2; ++i) {
        T[i] = value;   // top
        T[(n+1)*(n+2)+i] = value;   // bottom
        T[i*(n+2)] = value;   // left
        T[i*(n+2)+n+1] = value;   // right
    }
}


/// Prints our extremely minimal matrix to `os`.
void print_matrix(std::ostream& os, double* arr, int nrows, int ncols) {
    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            os << std::setw(2) << arr[idx] << " ";
        }
        os << std::endl;
    }
    os << std::endl;
}

/// Uses Gauss-Seidel iteration with a second-order central finite difference discretization.
/// Assume homogeneous Dirichlet boundary conditions for simplicity for now.
/// Parameters:     T: a matrix of size n+2 * n+2, with Dirichlet boundary conditions along the edges
///                 f: a matrix of size n * n, giving the forcing term for each mesh cell
///                 n: the number of non-boundary cells in each direction
int solve_stationary_heat_eqn(double* T, double* f, int n) {

    double error_threshold = 0.00001;
    double residual = 2*error_threshold;
    double h = 1.0/(n+1);
    double w = 1.0/(h * h);

    // Populate an initial guess inside domain
    for (int r=1; r<n+1; ++r) {
        for (int c=1; c<n+1; ++c) {
            T[r*(n+2)+c] = 1.0;
        }
    }

    int iters = 0;

    while (residual > error_threshold) {
        for (int r=1; r<n+1; ++r) {
            for (int c=1; c<n+1; ++c) {
                // T[r,c] = -forcing_fn(r,c)/wh + wx*(T[r,c-1] + T[r,c+1]) + wy*(T[r-1,c] + T[r+1,c]);
                T[r*(n+2)+c] = -f[(r-1)*(n)+(c-1)]/(4.0*w) + (T[r*(n+2)+c-1] + T[r*(n+2)+c+1] + T[(r-1)*(n+2)+c] + T[(r+1)*(n+2)+c])/4.0;
            }
        }
        print_matrix(std::cout, T, n+2, n+2);
        iters++;
        residual = 0;
        double cell_residual = 0;
        double sum_of_cell_residuals_squared = 0;
        for (int r=1; r<(n+2)-1; ++r) {
            for (int c=1; c<(n+2)-1; ++c) {
                cell_residual = -f[(r-1)*n+(c-1)] - 4*w*T[r*(n+2)+c] + w*(T[r*(n+2)+c-1] + T[r*(n+2)+c+1] + T[(r-1)*(n+2)+c] + T[(r+1)*(n+2)+c]);
                sum_of_cell_residuals_squared += cell_residual*cell_residual;
            }
        }
        residual = sqrt((1.0/(n*n)) * sum_of_cell_residuals_squared);
        std::cout << "Residual is " << residual << std::endl;
    }
    return iters;
}



int main() {
    constexpr size_t N = 7;

    double T[(N+2) * (N+2)];
    double f[N * N];

    zero_matrix(T, N+2, N+2);

    make_forcing_term(f, N);
    print_matrix(std::cout, f, N, N);

    make_boundary(T, N, 0);
    print_matrix(std::cout, T, N+2, N+2);

    solve_stationary_heat_eqn(T, f, N);
    print_matrix(std::cout, T, N+2, N+2);

    return 0;
}

