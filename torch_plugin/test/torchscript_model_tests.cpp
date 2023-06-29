/*
Copyright 2022-2023, Jefferson Science Associates, LLC.
Subject to the terms in the LICENSE file found in the top-level directory.
Authors: Nathan Brei (nbrei@jlab.org), Xinxin Mei (xmei@jlab.org)
*/

// Launch this test by  phasm-torch-plugin-tests -p <path-to-mfield-pt-model>

// Make sure to remove the CATCH_CONFIG_MAIN macro from all other test files in your project 
// to avoid the multiple definition error.
#define CATCH_CONFIG_RUNNER  // this is necessary because of own-defined main function
#include <catch.hpp>
#include <string>

#include <surrogate_builder.h>
#include "torchscript_model.h"

using namespace phasm;

std::string ptPath;  // global variable taken from command line input

int main(int argc, char* argv[]) {
    Catch::Session session; // There must be exactly one instance

    // Build a new parser on top of Catch2's
    using namespace Catch::clara;
    auto cli
    = session.cli()           // Get Catch2's command line parser
    | Opt( ptPath, "path" ) // bind variable to a new option, with a hint string
        ["-p"]["--path"]    // the option names it will respond to
        ("Location to the mfield torchscript model");  // description string for the help output

    // Now pass the new composite back to Catch2 so it uses that
    session.cli( cli );

    // Let Catch2 (using Clara) parse the command line
    int returnCode = session.applyCommandLine( argc, argv );
    if( returnCode != 0 ) // Indicates a command line error
        return returnCode;

    // std::cout << "Module path: " << ptPath << std::endl;

    return session.run();
}

/**
 * Test the input module is related to mfield example
*/
TEST_CASE("Module has 'mfield' in its name") {
    REQUIRE(ptPath.find("mfield") != std::string::npos);
}

namespace phasm::test::torchscript_model_tests {

/**
 * Test loading a torchscript model and do one round of inference on CPU.
*/
TEST_CASE("Load the module to CPU and do inference") {
    auto s = SurrogateBuilder()
        .set_model(std::make_shared<TorchscriptModel>(TorchscriptModel(ptPath)), true)
        .local_primitive<double>("x", phasm::IN)
        .local_primitive<double>("y", phasm::IN)
        .local_primitive<double>("z", phasm::IN)
        .local_primitive<double>("Bx", Direction::OUT)
        .local_primitive<double>("By", Direction::OUT)
        .local_primitive<double>("Bz", Direction::OUT)
        .finish();

    // Set up a simple function pretending to be a magnetic field map data
    double x = 1.0, y = 2.0, z = 3.0, Bx, By, Bz;
    s.bind_original_function([&]() { Bx = -x, By = -y; Bz = -z; });
    s.bind_all_callsite_vars(&x, &y, &z, &Bx, &By, &Bz);

    s.call_original_and_capture();
    REQUIRE(Bx == -1.0);
    REQUIRE(By == -2.0);
    REQUIRE(Bz == -3.0);

    // Inference results, a loose constraint
    s.call_model_and_capture();
    REQUIRE(Bx != -1.0);
    REQUIRE(By != -2.0);
    REQUIRE(Bz != -3.0);
}

TEST_CASE("Load the module to assigned device (GPU prefered) and do inference") {
    torch::Device device(torch::cuda::is_available()? torch::kCUDA : torch::kCPU);

    auto s = SurrogateBuilder()
        .set_model(std::make_shared<TorchscriptModel>(
            TorchscriptModel(ptPath, false, device)), 
            true
            )
        .local_primitive<double>("x", phasm::IN)
        .local_primitive<double>("y", phasm::IN)
        .local_primitive<double>("z", phasm::IN)
        .local_primitive<double>("Bx", Direction::OUT)
        .local_primitive<double>("By", Direction::OUT)
        .local_primitive<double>("Bz", Direction::OUT)
        .finish();

    double x = 1.0, y = 2.0, z = 3.0, Bx, By, Bz;
    s.bind_original_function([&]() { Bx = -x, By = -y; Bz = -z; });
    s.bind_all_callsite_vars(&x, &y, &z, &Bx, &By, &Bz);

    s.call_model_and_capture();
    REQUIRE(Bx != -1.0);
    REQUIRE(By != -2.0);
    REQUIRE(Bz != -3.0);
}

} // namespace phasm::test::torchscript_model_tests
