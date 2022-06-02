
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "surrogate.h"

#include <catch.hpp>
#include <iostream>

using namespace phasm;

namespace phasm::tests::capturing_tests {

// Helper functions for validating our results
template<typename T>
T get_captured_input(std::shared_ptr<Model> model, std::string param_name, size_t sample_index) {
    auto param = model->get_model_var(param_name);
    torch::Tensor result = param->training_inputs[sample_index];
    return *result.data_ptr<T>();
}

template<typename T>
T get_captured_output(std::shared_ptr<Model> model, std::string param_name, size_t sample_index) {
    auto param = model->get_model_var(param_name);
    torch::Tensor result = param->training_outputs[sample_index];
    return *result.data_ptr<T>();
}

int mult(int x, int y) {
    return x * y;
}

TEST_CASE("Capture int(int,int)") {

    int x, y, z;
    auto model = std::make_shared<Model>();
    model->add_var<int>("x", Direction::Input);
    model->add_var<int>("y", Direction::Input);
    model->add_var<int>("z", Direction::Output);

    auto surrogate = Surrogate([&]() { z = mult(x, y); }, model);
    surrogate.bind("x", &x);
    surrogate.bind("y", &y);
    surrogate.bind("z", &z);

    x = 3;
    y = 5;
    surrogate.call_original_and_capture();
    REQUIRE(z == 15);

    REQUIRE(get_captured_input<int>(model, "x", 0) == 3);
    REQUIRE(get_captured_input<int>(model, "y", 0) == 5);
    REQUIRE(get_captured_output<int>(model, "z", 0) == 15);
}

int mult_const(const int x, const int y) {
    return x * y;
}

TEST_CASE("Capture int(const int, const int)") {

    int x = 3, y = 5, z = 0;
    auto m = std::make_shared<Model>();
    m->add_var<int>("x", Direction::Input);
    m->add_var<int>("y", Direction::Input);
    m->add_var<int>("z", Direction::Output);

    auto surrogate = Surrogate([&]() { z = mult(x, y); }, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("y", &y);
    surrogate.bind<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(get_captured_input<int>(m, "x", 0) == 3);
    REQUIRE(get_captured_input<int>(m, "y", 0) == 5);
    REQUIRE(get_captured_output<int>(m, "z", 0) == 15);
}

int mult_with_ref(int &x, int &&y) {
    return x * y;
}

TEST_CASE("Capture int(int&,int&&)") {

    auto m = std::make_shared<Model>();
    m->add_var<int>("x", Direction::Input);
    m->add_var<int>("y", Direction::Input);
    m->add_var<int>("z", Direction::Output);

    int x = 3, y = 5, z = 0;
    auto surrogate = Surrogate([&]() { z = mult_with_ref(x, std::move(y)); }, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("y", &y);
    surrogate.bind<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(get_captured_input<int>(m, "x", 0) == 3);
    REQUIRE(get_captured_input<int>(m, "y", 0) == 5);
    REQUIRE(get_captured_output<int>(m, "z", 0) == 15);
}

int mult_with_out_param(int &x, int y) {
    int z = x * y;
    x = 22;
    return z;
}

TEST_CASE("Capture int(int&,int) [input and output]") {

    int x = 3, y = 5, z = 0;
    auto m = std::make_shared<Model>();
    m->add_var<int>("x", Direction::InputOutput);
    m->add_var<int>("y", Direction::Input);
    m->add_var<int>("z", Direction::Output);

    auto surrogate = Surrogate([&]() { z = mult_with_out_param(x, y); }, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("y", &y);
    surrogate.bind<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(x == 22);
    REQUIRE(get_captured_input<int>(m, "x", 0) == 3);
    REQUIRE(get_captured_input<int>(m, "y", 0) == 5);
    REQUIRE(get_captured_output<int>(m, "x", 0) == 22);
    REQUIRE(get_captured_output<int>(m, "z", 0) == 15);

    m->dump_captures_to_csv(std::cout);
}

int g = 22;

int mult_with_global(int x) {
    return x * g;
}

TEST_CASE("Capture int(int) [with global]") {

    int x = 5, z = 0;
    auto m = std::make_shared<Model>();
    m->add_var<int>("x", Direction::Input);
    m->add_var<int>("g", Direction::Input);
    m->add_var<int>("z", Direction::Output);

    auto surrogate = Surrogate([&]() { z = mult_with_global(x); }, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("g", &g);
    surrogate.bind<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 110);
    REQUIRE(get_captured_input<int>(m, "x", 0) == 5);
    REQUIRE(get_captured_input<int>(m, "g", 0) == 22);
    REQUIRE(get_captured_output<int>(m, "z", 0) == 110);
}

int no_args(void) {
    return 0;
}

TEST_CASE("Capture int() [with no args]") {

    int z = 22;
    auto m = std::make_shared<Model>();
    m->add_var<int>("z", Direction::Output);

    auto surrogate = Surrogate([&]() { z = no_args(); }, m);
    surrogate.bind<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 0);

    REQUIRE(get_captured_output<int>(m, "z", 0) == 0);
}

void return_void(int x) {
    x = x * x;
}

TEST_CASE("Capture void(int)") {

    int x = 3;
    auto m = std::make_shared<Model>();
    m->add_var<int>("x", Direction::Input);

    auto surrogate = Surrogate([&]() { return_void(x); }, m);
    surrogate.bind<int>("x", &x);

    surrogate.call_original_and_capture();
    REQUIRE(get_captured_input<int>(m, "x", 0) == 3);
}
} // namespace phasm::tests::capturing_tests

