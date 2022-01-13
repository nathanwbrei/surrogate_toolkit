
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "pytorch_model.h"

#include "surrogate.h"

double square(double x);

TEST_CASE("Can we build against pytorch at all?") {

    auto m = std::make_shared<PyTorchModel>();
    m->input<double>("x", {-3,3});
    m->output<double>("y");

    double x, y;
    auto s = Surrogate([&](){y = square(x);}, m);
    s.bind_input("x", &x);
    s.bind_output("y", &y);

    for (int i = -3; i<3; ++i) {
        // x =
	// s.call_original_and_capture();

    }
}
