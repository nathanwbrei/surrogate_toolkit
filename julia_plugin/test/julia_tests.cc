
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <julia_model.h>

TEST_CASE("Hello world") {
    
    phasm::JuliaModel model("TestModel.jl");
    model.initialize();
    //model.add_model_vars()
    
    REQUIRE(true == true);
}

