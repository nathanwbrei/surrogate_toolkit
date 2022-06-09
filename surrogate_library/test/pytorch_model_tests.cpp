
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "feedforward_model.h"

#include "surrogate.h"

using namespace phasm;
namespace phasm::test::pytorch_model_tests {

double square(double x) {
    return x*x;
}

TEST_CASE("Can we build against pytorch at all?") {


    auto s = Surrogate();
    auto m = std::make_shared<FeedForwardModel>();
    s.add_var<double>("x", new Primitive<double>(), "x", Direction::IN);
    s.add_var<double>("y", Direction::OUT);
    s.set_model(m);
    m->add_model_vars(s.get_model_vars());

    double x, y;
    s.bind_locals_to_original_function([&]() { y = square(x); });
    s.bind("x", &x);
    s.bind("y", &y);

    for (int i = -3; i < 3; ++i) {
        // x =
        // s.call_original_and_capture();

    }
}
} // namespace phasm::test::pytorch_model_tests