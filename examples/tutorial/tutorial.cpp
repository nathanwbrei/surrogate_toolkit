
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <surrogate_builder.h>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

/* This tutorial describes the key features of libphasm. These allow the user to introduce a neural net surrogate model
 * for an arbitrary function, capture training data, create new training data via sampling, train the model in
 * place, etc. Although the PHASM project also includes performance analysis and memory tracing functionality, they will
 * not be covered here.
 */

/* Here is the simplest possible use of libphasm. Given some arbitrary (in reality quite expensive) function,
 * we'd like to create a _surrogate_ which mimics it to a desired order of accuracy. For illustration purposes we'll
 * use a simple polynomial.
 * */

double f(double x, double y, double z) {
    return 3*x*x + 2*y + z;
}

// We create a surrogate model for f like so:

phasm::Surrogate f_surrogate = phasm::SurrogateBuilder()                  // [1]
        .set_model("phasm-torch-plugin", "")                              // [2]
        .local_primitive<double>("x", phasm::IN)                          // [3]
        .local_primitive<double>("y", phasm::IN)
        .local_primitive<double>("z", phasm::IN)
        .local_primitive<double>("returns", phasm::OUT)
        .finish();                                                        // [4]

/* Let's unpack this piece by piece.
 *
 * [1] PHASM's main abstraction for a surrogate model is called `Surrogate`. Because we are
 *     wrapping the original function and always calling the wrapper, we only need one Surrogate
 *     object for f_wrapper, and it needs to last for the lifetime of the program,
 *     so we make it global and/or static. To configure a Surrogate, we use the `SurrogateBuilder`,
 *     which provides a fluent interface.
 *
 * [2] The first thing we need to tell the SurrogateBuilder is which model to use. Here we are
 *     using FeedForwardModel, which gives us the simplest possible neural net architecture using
 *     PyTorch as the backend. You can configure the number and size of the hidden layers using
 *     optional constructor arguments; the model automatically sizes its input and output layers
 *     to match the size of the input and output data. Generally, however, we expect users
 *     to design a model inside JupyterLab and export it using TorchScript. PHASM conveniently
 *     provides `TorchScriptModel` for this purpose. You can also write your own Model subclasses.
 *
 * [3] The next thing the Surrogate needs to know about is all of its inputs and outputs. Here we
 *     declare that the Surrogate should accept as inputs 3 doubles from the local scope, and
 *     as outputs one double. So far this is comparable other surrogate modelling toolkits, but
 *     just keep reading!
 *
 * [4] Once we've passed the SurrogateBuilder all the information it needs, we call `finish()`,
 *     which tells it to construct the Surrogate and negotiate the input and output data sizes with
 *     the Model. We've configured the Surrogate! (Note that because we declared the Surrogate to be
 *     static, we bypassed the lazy initialization/double-checked locking problem, so the initialization
 *     is thread-safe.)
 */

// Here is what we can now do with the surrogate:
TEST_CASE("Calling f_surrogate") {

    double x, y, z, result;                                       // [5]

    f_surrogate
        .bind_original_function([&](){ result = f(x,y,z); })      // [6]
        .bind_all_callsite_vars(&x, &y, &z, &result);                      // [7]

    x = 1; y = 1; z = 1;
    f_surrogate.call_original_and_capture();
    REQUIRE(result == 6);

    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            for (int k=0; k<5; ++k) {

                x = i; y = j; z = k;
                f_surrogate.call_original_and_capture();
                REQUIRE(result == 3*x*x + 2*y + z);
            }
        }
    }

    f_surrogate.get_model()->train_from_captures();                // [9]

    x = 1; y = 1; z = 1;
    f_surrogate.call_model();                                      // [10]
    // REQUIRE(result == 6);
    std::cout << "Model expected 6, actually returned " << result << std::endl;

}

// One thing that makes general surrogate models harder in C/C++ compared to other
// languages is value semantics. Essentially, the


// In a real-world codebase, we usually don't want to modify every call site for f, or
// hard-code actions such as "call_original_and_capture()". Instead, we
// want to create a wrapper function for f and make the surrogate machinery transparent
// to the rest of the program. We want to be able to specify the action our surrogate
// model takes via a side-channel. In this case, we


// We can create a wrapper function for f like so:

double f_wrapper(double x, double y, double z) {
    double result = 0.0;
    f_surrogate.bind_original_function([&](){ result = f(x,y,z); })      // [5]
               .bind_all_callsite_vars(&x, &y, &z, &result)              // [6]
               .call();                                                  // [7]
    return result;
}

TEST_CASE("Call f_wrapper") {

    f_surrogate.set_callmode(phasm::CallMode::CaptureAndDump);

    double result = f_wrapper(1,2,3);
    std::cout << "f(1,2,3) = " << f(1,2,3) << "; f_wrapper(1,2,3) = " << result << std::endl;
    result = f_wrapper(2,2,3);
    std::cout << "f(2,2,3) = " << f(2,2,3) << "; f_wrapper(2,2,3) = " << result << std::endl;
    result = f_wrapper(3,2,3);
    std::cout << "f(3,2,3) = " << f(3,2,3) << "; f_wrapper(3,2,3) = " << result << std::endl;
    result = f_wrapper(4,2,3);
    std::cout << "f(4,2,3) = " << f(4,2,3) << "; f_wrapper(4,2,3) = " << result << std::endl;

    // The results should be identical because PHASM will redirect calls to f_wrapper back to f.
    // Meanwhile, because the call mode is CaptureAndDump, PHASM will capture the inputs and outputs for each call.
    // When the program exits, PHASM will dump the captures to CSV.
}


/*
 * [5] For our Surrogate to call the original function f, we need to _bind_ the input and output parameters to f.
 *     We pass the Surrogate a lambda function with type signature `void(void)`. Note that we have to declare a
 *     variable on the stack to use as the return value. If you are wondering why we do this on every call as
 *     opposed to once during startup, it is because the input parameters might be references, which would otherwise
 *     capture incorrectly.
 *
 * [6] For our Surrogate to use the Model, we need to bind the input and output parameters to the model. This is
 *     fundamentally more interesting, because the Model expects tensors containing primitives, not arbitrary C/C++
 *     types. For this simple case, all we have to do is call `bind_all_callsite_vars()` passing in _pointers_ to
 *     the parameters in the order they were declared to the SurrogateBuilder.
 *
 * [7] Finally, we can use our Surrogate. The code shown here will call the original function, capture both the
 *     inputs and outputs as tensors, and save them. We can later dump the data to CSV or use it to train the model
 *     directly.
 * */


template <typename T>
void print_matrix(std::ostream& os, T* arr, int nrows, int ncols) {
    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            os << std::setw(2) << arr[idx] << " ";
        }
        os << std::endl;
    }
    os << std::endl;
}

template <typename T>
void fill_matrix(T* arr, int nrows, int ncols) {
    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            size_t idx = row*ncols + col;
            arr[idx] = idx;
        }
    }
}

template <typename T>
void zero_top_right(T* arr, int nrows, int ncols) {

    for (int row=0; row<nrows; ++row) {
        for (int col=0; col<ncols; ++col) {
            if (row < col) {
                size_t idx = row*ncols + col;
                arr[idx] = 0;
            }
        }
    }
}


TEST_CASE("Surrogates with array input data") {

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


    using namespace phasm;
    Surrogate zero_top_right_surrogate = SurrogateBuilder()
            .set_model("phasm-torch-plugin", "")
            .local_primitive<float>("m", INOUT, {4, 5})
            .finish();

    // note we don't include nrows and ncols as model params because they are
    // implicitly included in the shape of parameter "m". Also note that this
    // implies that our model is constrained to use

    zero_top_right_surrogate
        .bind_original_function([&](){ return zero_top_right(matrix, 4, 5); })
        .bind_all_callsite_vars(matrix);

    fill_matrix(matrix, 4, 5);
    zero_top_right_surrogate.call_model();

    std::cout << "Buffer after running the surrogate model:" << std::endl;
    print_matrix(std::cout, matrix, 4, 5);


}

// Global variables
// Arrays of structs of structs of arrays of data
