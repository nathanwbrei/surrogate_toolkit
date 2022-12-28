
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "surrogate_builder.h"
#include "model.h"
#include <catch.hpp>
#include <iostream>

using namespace phasm;

namespace phasm::tests::capturing_tests {

// Helper functions for validating our results
template<typename T>
T get_captured_input(std::shared_ptr<Model> model, std::string param_name, size_t sample_index) {
    auto param = model->get_model_var(param_name);
    tensor result = param->training_inputs[sample_index];
    return *result.get<T>();
}

template<typename T>
T get_captured_output(std::shared_ptr<Model> model, std::string param_name, size_t sample_index) {
    auto param = model->get_model_var(param_name);
    tensor result = param->training_outputs[sample_index];
    return *result.get<T>();
}

int mult(int x, int y) {
    return x * y;
}

TEST_CASE("Capture int(int,int)") {

    int x, y, z;
    auto model = std::make_shared<Model>();

    Surrogate surrogate = SurrogateBuilder()
        .set_model(model)
        .local_primitive<int>("x", IN)
        .local_primitive<int>("y", IN)
        .local_primitive<int>("z", OUT)
        .finish();

    surrogate.bind_original_function([&]() { z = mult(x, y); });
    surrogate.bind_callsite_var("x", &x);
    surrogate.bind_callsite_var("y", &y);
    surrogate.bind_callsite_var("z", &z);

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
    auto surrogate = SurrogateBuilder()
            .set_model(m)
            .local_primitive<int>("x", IN)
            .local_primitive<int>("y", IN)
            .local_primitive<int>("z", OUT)
            .finish();

    surrogate.bind_original_function([&]() { z = mult_const(x, y); });
    surrogate.bind_callsite_var<int>("x", &x);
    surrogate.bind_callsite_var<int>("y", &y);
    surrogate.bind_callsite_var<int>("z", &z);

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
    auto surrogate = SurrogateBuilder()
            .set_model(m)
            .local_primitive<int>("x", IN)
            .local_primitive<int>("y", IN)
            .local_primitive<int>("z", OUT)
            .finish();

    int x = 3, y = 5, z = 0;
    surrogate.bind_original_function([&]() { z = mult_with_ref(x, std::move(y)); });
    surrogate.bind_callsite_var<int>("x", &x);
    surrogate.bind_callsite_var<int>("y", &y);
    surrogate.bind_callsite_var<int>("z", &z);

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
    auto surrogate = SurrogateBuilder()
            .set_model(m)
            .local_primitive<int>("x", Direction::INOUT)
            .local_primitive<int>("y", Direction::IN)
            .local_primitive<int>("z", Direction::OUT)
            .finish();

    surrogate.bind_original_function([&]() { z = mult_with_out_param(x, y); });
    surrogate.bind_callsite_var<int>("x", &x);
    surrogate.bind_callsite_var<int>("y", &y);
    surrogate.bind_callsite_var<int>("z", &z);

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
    auto surrogate = SurrogateBuilder()
            .set_model(m)
            .local_primitive<int>("x", Direction::IN)
            .local_primitive<int>("g", Direction::IN)
            .local_primitive<int>("z", Direction::OUT)
            .finish();

    surrogate.bind_original_function([&]() { z = mult_with_global(x); });
    surrogate.bind_callsite_var<int>("x", &x);
    surrogate.bind_callsite_var<int>("g", &g);
    surrogate.bind_callsite_var<int>("z", &z);

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
    auto surrogate = SurrogateBuilder()
            .set_model(m)
            .local_primitive<int>("z", Direction::OUT)
            .finish();

    surrogate.bind_original_function([&]() { z = no_args(); });
    surrogate.bind_callsite_var<int>("z", &z);

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
    auto surrogate = SurrogateBuilder()
            .set_model(m)
            .local_primitive<int>("x", Direction::IN)
            .finish();

    surrogate.bind_original_function([&]() { return_void(x); });
    surrogate.bind_callsite_var<int>("x", &x);

    surrogate.call_original_and_capture();
    REQUIRE(get_captured_input<int>(m, "x", 0) == 3);
}


} // namespace phasm::tests::capturing_tests

