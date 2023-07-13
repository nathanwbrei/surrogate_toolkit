
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <julia_model.h>
#include <surrogate_builder.h>


namespace phasm_julia_tests {

double square(double x) { return x * x; }

TEST_CASE("Scalar-valued result") {
    
    auto model = std::make_shared<phasm::JuliaModel>("TestModel.jl");

    phasm::Surrogate f_surrogate = phasm::SurrogateBuilder()
        .set_model(model)
        .local_primitive<double>("x", phasm::INOUT)
        .local_primitive<double>("f", phasm::OUT)
        .finish();

    double x = 2.0;
    double f = 0.0;
    std::cout << "Before call_model(): x = " << x << ", f = " << f << std::endl;

    f_surrogate
        .bind_original_function([&](){ f = square(x); })
        .bind_all_callsite_vars(&x, &f)
        .call_model();

    std::cout << "After call_model(): x = " << x << ", f = " << f << std::endl;
    REQUIRE(f != 0.0);
}

} // namespace phasm_julia_tests