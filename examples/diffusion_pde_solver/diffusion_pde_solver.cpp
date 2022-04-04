
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <iomanip>

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
void zero_top_right(float* arr, size_t nrows, size_t ncols) {

    for (size_t row=0; row<nrows; ++row) {
        for (size_t col=0; col<ncols; ++col) {
            if (row < col) {
                size_t idx = row*ncols + col;
                arr[idx] = 0;
            }
        }
    }
}

/// Prints our extremely minimal matrix to `os`.
void print_matrix(std::ostream& os, float* arr, size_t nrows, size_t ncols) {
    for (size_t row=0; row<nrows; ++row) {
        for (size_t col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            os << std::setw(2) << arr[idx] << " ";
        }
        os << std::endl;
    }
    os << std::endl;
}

int main() {
    float matrix[] = { 1,  2,  3,  4,  5,
                       6,  7,  8,  9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20 };

    std::cout << "Buffer before running the original function: " << std::endl;
    print_matrix(std::cout, matrix, 4, 5);

    zero_top_right(matrix, 4, 5);

    std::cout << "Buffer after running the original function: " << std::endl;
    print_matrix(std::cout, matrix, 4, 5);

}

