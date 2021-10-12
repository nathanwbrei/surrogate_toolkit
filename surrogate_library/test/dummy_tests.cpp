
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "surrogate_builder.h"

TEST_CASE("Dummy test cases for surrogate tool") {
    hello_from_surrogate_library();
    SECTION("JustForFun") {
        REQUIRE(1 == 1);
    }
    SECTION("AndAnother") {
        REQUIRE(2 == 2);
    }
}


