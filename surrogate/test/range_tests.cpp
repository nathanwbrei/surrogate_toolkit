
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "range.h"

#include <catch.hpp>
#include <iostream>

using namespace phasm;

template <typename T>
phasm::tensor scalar_to_tensor(T scalar) {
    return tensor(&scalar, 1);
}

TEST_CASE("Interval range") {
    auto x = Range(scalar_to_tensor(-5), scalar_to_tensor(5));
    REQUIRE(x.contains(scalar_to_tensor(-5)));
    REQUIRE(x.contains(scalar_to_tensor(5)));
    REQUIRE(x.contains(scalar_to_tensor(0)));
    REQUIRE(x.contains(scalar_to_tensor(2)));
    REQUIRE(x.contains(scalar_to_tensor(-10)) == false);
    REQUIRE(x.contains(scalar_to_tensor(10)) == false);
}

TEST_CASE("FiniteSet range") {
    auto x = Range({scalar_to_tensor(1),scalar_to_tensor(2),scalar_to_tensor(3),scalar_to_tensor(4)});
    REQUIRE(x.contains(scalar_to_tensor(1)));
    REQUIRE(!x.contains(scalar_to_tensor(10)));
}

TEST_CASE("Range capturing") {
    Range rf(scalar_to_tensor(100),scalar_to_tensor(0));
    std::vector<int> samples = {7,0,3,7,9,144,7,0};
    for (int x : samples) {
        rf.capture(scalar_to_tensor(x));
    }
    REQUIRE(rf.lower_bound_inclusive == scalar_to_tensor(0));
    REQUIRE(rf.upper_bound_inclusive == scalar_to_tensor(144));
    // REQUIRE(rf.distribution.size() == 5);
    rf.report(std::cout);

}
