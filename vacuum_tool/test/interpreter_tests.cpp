
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "vacuum_interpreter.hpp"

using namespace phasm::vacuumtool;

TEST_CASE("Create interpreter") {
    VacuumInterpreter sut(1, {"main","target","f"});

}