
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "pytorch_model.h"

#include "surrogate_builder.h"

double square(double x);

TEST_CASE("Can we build against pytorch at all?") {
    PyTorchModel m;

    double x, y;
    auto s = Surrogate([&](){y = square(x);});
    s.input("x", &x, {-3,3});
    s.output("y", &x);

    for (int i = -3; i<3; ++i) {
        // x =
	// s.call_original_and_capture();

    }






}
