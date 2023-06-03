
// Copyright 2022-2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.
// Authors: Nathan Brei (nbrei@jlab.org)

#include <catch.hpp>

#include <surrogate_builder.h>
#include "torchscript_model.h"

using namespace phasm;
namespace phasm::test::torchscript_model_tests {

TEST_CASE("TorchScriptModelTests") {

    // Note that we shouldn't use set_model("phasm-torch-plugin","") here because of the one-definition rule!!!
    auto s = SurrogateBuilder()
            .set_model(std::make_shared<TorchscriptModel>(""))
            .local_primitive<double>("p", IN)
            .local_primitive<double>("theta", IN)
            .local_primitive<double>("E_abs", OUT)
            .local_primitive<double>("E_gap", OUT)
            .finish();

    // Set up a simple function pretending to be a sampling calorimeter simulation
    double p, theta, E_abs, E_gap;
    s.bind_original_function([&]() { E_abs = p*p; E_gap = p+theta; });
    s.bind_all_callsite_vars(&p, &theta, &E_abs, &E_gap);

    p = 3.0;
    theta = 2.0;
    s.call_original_and_capture();
    REQUIRE(E_abs == 9.0);
    REQUIRE(E_gap == 5.0);

    p = 5.0;
    theta = 3.0;
    s.call_model();
    // The model isn't trained, so it updates the final values with garbage
    // The important thing is simply that it updates the values
    REQUIRE(E_abs != 9.0);
    REQUIRE(E_gap != 5.0);
}
} // namespace phasm::test::torchscript_model_tests