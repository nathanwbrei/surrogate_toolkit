
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <dwarf.h>
#include <libdwarf.h>
#include "utils.hpp"


TEST_CASE("Demangling a string") {
    std::string name = "__Z7target1id";   // macOS does this!
    auto result = demangle(name);
    REQUIRE(result == "target1(int, double)");
}


