
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>

#include "surrogate_builder.h"

using namespace phasm;
namespace phasm::test::pytorch_model_tests {

double square(double x) {
    return x*x;
}

TEST_CASE("Can we build against pytorch at all?") {


    auto s = SurrogateBuilder()
            .set_model("phasm-torch-plugin", "")
            .local_primitive<double>("x", IN)
            .local_primitive<double>("y", OUT)
            .finish();

    double x, y;
    s.bind_original_function([&]() { y = square(x); });
    s.bind_callsite_var("x", &x);
    s.bind_callsite_var("y", &y);

    for (int i = -3; i < 3; ++i) {
        // x =
        // s.call_original_and_capture();

    }
}
} // namespace phasm::test::pytorch_model_tests