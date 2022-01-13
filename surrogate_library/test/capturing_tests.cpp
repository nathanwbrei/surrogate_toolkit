
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "surrogate_builder.h"

int mult(int x, int y) {
    return x * y;
}

TEST_CASE("Capture int(int,int)") {

    int x,y,z;
    auto model = std::make_shared<Model>();
    model->input<int>("x");
    model->input<int>("y");
    model->output<int>("z");

    auto surrogate = Surrogate([&](){z = mult(x,y);}, model);
    surrogate.bind_input("x", &x);
    surrogate.bind_input("y", &y);
    surrogate.bind_output("z", &z);

    x = 3; y = 5;
    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
    REQUIRE(surrogate.get_captured_input<int>(0, 1) == 5);
    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 15);
}

int mult_const(const int x, const int y) {
    return x * y;
}

TEST_CASE("Capture int(const int, const int)") {

    int x = 3,y = 5,z = 0;
    auto m = std::make_shared<Model>();
    m->input<int>("x");
    m->input<int>("y");
    m->output<int>("z");

    auto surrogate = Surrogate([&](){z = mult(x,y);}, m);
    surrogate.bind_input<int>("x", &x);
    surrogate.bind_input<int>("y", &y);
    surrogate.bind_output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
    REQUIRE(surrogate.get_captured_input<int>(0, 1) == 5);
    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 15);
}

int mult_with_ref(int& x, int&& y) {
    return x * y;
}

TEST_CASE("Capture int(int&,int&&)") {

    auto m = std::make_shared<Model>();
    m->input<int>("x");
    m->input<int>("y");
    m->output<int>("z");

    int x = 3, y = 5, z = 0;
    auto surrogate = Surrogate([&](){z = mult_with_ref(x,std::move(y));}, m);
    surrogate.bind_input<int>("x", &x);
    surrogate.bind_input<int>("y", &y);
    surrogate.bind_output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
    REQUIRE(surrogate.get_captured_input<int>(0, 1) == 5);
    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 15);
}

int mult_with_out_param(int& x, int y) {
    int z = x*y;
    x = 22;
    return z;
}

TEST_CASE("Capture int(int&,int) [input and output]") {

    int x=3, y=5, z=0;
    auto m = std::make_shared<Model>();
    m->input_output<int>("x");
    m->input<int>("y");
    m->output<int>("z");

    auto surrogate = Surrogate([&](){z = mult_with_out_param(x,y);}, m);
    surrogate.bind_input_output<int>("x", &x);
    surrogate.bind_input<int>("y", &y);
    surrogate.bind_output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(x == 22);
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
    REQUIRE(surrogate.get_captured_input<int>(0, 1) == 5);
    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 22);
    REQUIRE(surrogate.get_captured_output<int>(0, 1) == 15);
}

int g = 22;
int mult_with_global(int x) {
    return x * g;
}

TEST_CASE("Capture int(int) [with global]") {

    int x = 5, z = 0;
    auto m = std::make_shared<Model>();
    m->input<int>("x");
    m->input<int>("g");
    m->output<int>("z");

    auto surrogate = Surrogate([&](){z = mult_with_global(x);}, m);
    surrogate.bind_input<int>("x", &x);
    surrogate.bind_input<int>("g", &g);
    surrogate.bind_output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 110);
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 5);
    REQUIRE(surrogate.get_captured_input<int>(0, 1) == 22);
    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 110);
}

int no_args(void) {
    return 0;
}

TEST_CASE("Capture int() [with no args]") {

    int z = 22;
    auto m = std::make_shared<Model>();
    m->output<int>("z");

    auto surrogate = Surrogate([&](){z = no_args();}, m);
    surrogate.bind_output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 0);

    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 0);
}

void return_void(int x) {
    x = x*x;
}

TEST_CASE("Capture void(int)") {

    int x = 3;
    auto m = std::make_shared<Model>();
    m->input<int>("x");

    auto surrogate = Surrogate([&](){return_void(x);}, m);
    surrogate.bind_input<int>("x", &x);

    surrogate.call_original_and_capture();
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
}


