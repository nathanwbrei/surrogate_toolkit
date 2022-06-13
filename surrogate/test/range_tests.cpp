
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "range.h"

#include <catch.hpp>
#include <iostream>

using namespace phasm;

TEST_CASE("Interval range") {
    auto x = Range<int>(-5, 5);
    REQUIRE(x.contains(-5));
    REQUIRE(x.contains(5));
    REQUIRE(x.contains(0));
    REQUIRE(x.contains(2));
    REQUIRE(x.contains(-10) == false);
    REQUIRE(x.contains(10) == false);
}

TEST_CASE("FiniteSet range") {
    auto x = Range<int>({1,2,3,4});
    REQUIRE(x.contains(1));
    REQUIRE(!x.contains(10));
}

TEST_CASE("Range capturing") {
    Range<int> rf(100,0);
    std::vector<int> samples = {7,0,3,7,9,144,7,0};
    for (int x : samples) {
        rf.capture(x);
    }
    REQUIRE(rf.lower_bound_inclusive == 0);
    REQUIRE(rf.upper_bound_inclusive == 144);
    REQUIRE(rf.distribution.size() == 5);
    rf.report(std::cout);

}
