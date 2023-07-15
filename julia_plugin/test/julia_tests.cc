
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <julia_model.h>
#include <surrogate_builder.h>


namespace phasm_julia_tests {

double plusplus(double& x) { return x++; }

TEST_CASE("Scalar-shaped tensor reading and writing") {
    
    auto model = std::make_shared<phasm::JuliaModel>("ScalarModel.jl");

    phasm::Surrogate f_surrogate = phasm::SurrogateBuilder()
        .set_model(model)
        .local_primitive<double>("x", phasm::INOUT)
        .local_primitive<double>("f", phasm::OUT)
        .finish();

    double x = 2.0;
    double f = 0.0;
    std::cout << "From C++ caller: Before call_model(): x = " << x << ", f = " << f << std::endl;

    f_surrogate
        .bind_original_function([&](){ f = plusplus(x); })
        .bind_all_callsite_vars(&x, &f)
        .call_model();

    std::cout << "From C++ caller: After call_model(): x = " << x << ", f = " << f << std::endl;
    REQUIRE(x == 22.0);
    REQUIRE(f == 33.0);
}

double square2x3(double* x) { 
    // square each entry in the 2x3 matrix in-place, return the sum of squares
    double sum = 0;
    for (int i=0; i<6; ++i) {
        x[i] *= x[i];
        sum += x[i];
    }
    return sum;
}

TEST_CASE("Oddly-shaped tensor reading and writing") {
    
    auto model = std::make_shared<phasm::JuliaModel>("OddModel.jl");

    phasm::Surrogate f_surrogate = phasm::SurrogateBuilder()
        .set_model(model)
        .local_primitive<double>("mat", phasm::INOUT, {2, 3})
        .local_primitive<double>("sum", phasm::OUT)
        .finish();

    double matrix[6] = { 2.0, 3.0, 7.0, 14.0, 9.0, 1.0};  // This is actually a 2x3 matrix as far as PHASM and Julia are concerned
    double sum = 0.0;
    // std::cout << "Before call_model(): x = " << x << ", f = " << f << std::endl;

    f_surrogate
        .bind_original_function([&](){ sum = square2x3(matrix); })
        .bind_all_callsite_vars(&matrix, &sum)
        .call_model();

    // std::cout << "After call_model(): x = " << x << ", f = " << f << std::endl;
    REQUIRE(matrix[0] == 4.0);
    REQUIRE(matrix[1] == 9.0);
    REQUIRE(matrix[2] == 49.0);
    REQUIRE(matrix[3] == 196.0);
    REQUIRE(matrix[4] == 81.0);
    REQUIRE(matrix[5] == 1.0);
    REQUIRE(sum == 4.0 + 9.0 + 49.0 + 196.0 + 81.0 + 1.0);
}


} // namespace phasm_julia_tests
