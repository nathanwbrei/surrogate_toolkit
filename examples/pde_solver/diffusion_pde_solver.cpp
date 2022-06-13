
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <iomanip>
#include <model.h>
#include <surrogate.h>
#include "feedforward_model.h"

/// Modify a 2D array by setting each cell in the top right to zero.
/// The idea is that it should be pretty easy to train a neural net to
/// do this, as long as we keep the size fixed. In this case, `arr` is
/// both an input and an output, and the `nrows` and `ncols` are not
/// needed as inputs to the neural net at all because they are encoded
/// in the tensor shape.
/// One more interesting question is how to make this work without
/// restricting us to one exact size. It might be possible using
/// RaggedTensors, or we might want to rescale to fixed size, feed it to
/// the neural net, and then rescale back. For now, however, this also
/// serves as a good demonstration of using `Range` to constrain our
/// surrogate model to delegate to the surrogate model only when the
/// matrix size is correct.
void zero_top_right(float* arr, int nrows, int ncols) {

    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            if (row < col) {
                size_t idx = row*ncols + col;
                arr[idx] = 0;
            }
        }
    }
}

/// Prints our extremely minimal matrix to `os`.
void print_matrix(std::ostream& os, float* arr, int nrows, int ncols) {
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
/// Parameters:     temperature_matrix
///                 nrows
///                 ncols


int solve_stationary_heat_eqn(float* temperature_matrix, int nrows, int ncols,
                               std::function<float(float, float)> forcing_fn
                               ) {

    float* T = temperature_matrix;
    double error_threshold = 0.0001;
    double residual = 2*error_threshold;
    double hx = 1.0/(ncols+1);
    double hy = 1.0/(nrows+1);
    double wx = 1.0/(hx * hx);
    double wy = 1.0/(hy * hy);
    double wh = 2*(wx+wy);   // Usually 4/(h^2)
    int iters = 0;

    while (residual > error_threshold) {
        for (int r=1; r<nrows-1; ++r) {
            for (int c=1; c<ncols-1; ++c) {
                // T[r,c] = -forcing_fn(r,c)/wh + wx*(T[r,c-1] + T[r,c+1]) + wy*(T[r-1,c] + T[r+1,c]);
                T[r*ncols+c] = -forcing_fn(r,c)/wh + wx*(T[r*ncols+c-1] + T[r*ncols+c+1]) + wy*(T[(r-1)*ncols+c] + T[(r+1)*ncols+c]);
            }
        }
        print_matrix(std::cout, temperature_matrix, nrows, ncols);
        iters++;
        residual = 0;
        double cell_residual = 0;
        double sum_of_cell_residuals_squared = 0;
        for (int r=1; r<nrows-1; ++r) {
            for (int c=1; c<ncols-1; ++c) {
                cell_residual = -forcing_fn(r,c) - wh*T[r*ncols+c] + wx*(T[r*ncols+c-1] + T[r*ncols+c+1]) + wy*(T[(r-1)*ncols+c] + T[(r+1)*ncols+c]);
                sum_of_cell_residuals_squared += cell_residual*cell_residual;
            }
        }
        residual = sqrt((1.0/(nrows+ncols)) * sum_of_cell_residuals_squared);
    }
    return iters;
}


void fill_matrix(float* arr, int nrows, int ncols) {
    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            arr[idx] = idx;
        }
    }
}

void zero_matrix(float* arr, int nrows, int ncols) {
    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            arr[idx] = 0;
        }
    }
}

#define PI 3.141592


int main() {
    /*
    float matrix[] = { 1,  2,  3,  4,  5,
                       6,  7,  8,  9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20 };

    std::cout << "Buffer before running the original function: " << std::endl;
    print_matrix(std::cout, matrix, 4, 5);

    zero_top_right(matrix, 4, 5);

    std::cout << "Buffer after running the original function: " << std::endl;
    print_matrix(std::cout, matrix, 4, 5);


    std::cout << "Buffer after being reset:" << std::endl;
    fill_matrix(matrix, 4, 5);
    print_matrix(std::cout, matrix, 4, 5);


    auto model = std::make_shared<FeedForwardModel>();
    model->input_output("arr", new optics::PrimitiveArray<float>({4,5}));
    model->input("nrows", new optics::Primitive<int>());
    model->input("ncols", new optics::Primitive<int>());
    model->initialize();

    int nrows = 4;
    int ncols = 5;
    Surrogate surrogate([&](){ zero_top_right(matrix, nrows, ncols); }, model);
    surrogate.bind_input_output("arr", matrix);
    surrogate.bind_input("nrows", &nrows);
    surrogate.bind_input("ncols", &ncols);

    surrogate.call_original_and_capture();
    model->train_from_captures();

    fill_matrix(matrix, 4, 5);
    surrogate.call_model();

    std::cout << "Buffer after running the surrogate model:" << std::endl;
    print_matrix(std::cout, matrix, 4, 5);

     */
    float temperature[10*10];
    zero_matrix(temperature, 10, 10);
    auto forcing_fn = [](float x, float y) {return -2*PI*PI*sin(PI*x)*sin(PI*y);};
    int iters = solve_stationary_heat_eqn(temperature, 10, 10, forcing_fn);
    std::cout << "Calculated solution in " << iters << " iters." << std::endl;
    print_matrix(std::cout, temperature, 10, 10);
}

