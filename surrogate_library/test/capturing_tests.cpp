
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "surrogate.h"

#include <catch.hpp>
#include <iostream>

int mult(int x, int y) {
    return x * y;
}

TEST_CASE("Capture int(int,int)") {

    int x,y,z;
    auto model = std::make_shared<Model>();
    model->add_input<int>("x");
    model->add_input<int>("y");
    model->add_output<int>("z");

    auto surrogate = Surrogate([&](){z = mult(x,y);}, model);
    surrogate.bind("x", &x);
    surrogate.bind("y", &y);
    surrogate.bind("z", &z);

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
    m->add_input<int>("x");
    m->add_input<int>("y");
    m->add_output<int>("z");

    auto surrogate = Surrogate([&](){z = mult(x,y);}, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("y", &y);
    surrogate.bind<int>("z", &z);

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
    m->add_input<int>("x");
    m->add_input<int>("y");
    m->add_output<int>("z");

    int x = 3, y = 5, z = 0;
    auto surrogate = Surrogate([&](){z = mult_with_ref(x,std::move(y));}, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("y", &y);
    surrogate.bind<int>("z", &z);

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
    m->add_input_output<int>("x");
    m->add_input<int>("y");
    m->add_output<int>("z");

    auto surrogate = Surrogate([&](){z = mult_with_out_param(x,y);}, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("y", &y);
    surrogate.bind<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(x == 22);
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
    REQUIRE(surrogate.get_captured_input<int>(0, 1) == 5);
    REQUIRE(surrogate.get_captured_output<int>(0, 0) == 22);
    REQUIRE(surrogate.get_captured_output<int>(0, 1) == 15);

    m->dump_captures_to_csv(std::cout);
}

int g = 22;
int mult_with_global(int x) {
    return x * g;
}

TEST_CASE("Capture int(int) [with global]") {

    int x = 5, z = 0;
    auto m = std::make_shared<Model>();
    m->add_input<int>("x");
    m->add_input<int>("g");
    m->add_output<int>("z");

    auto surrogate = Surrogate([&](){z = mult_with_global(x);}, m);
    surrogate.bind<int>("x", &x);
    surrogate.bind<int>("g", &g);
    surrogate.bind<int>("z", &z);

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
    m->add_output<int>("z");

    auto surrogate = Surrogate([&](){z = no_args();}, m);
    surrogate.bind<int>("z", &z);

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
    m->add_input<int>("x");

    auto surrogate = Surrogate([&](){return_void(x);}, m);
    surrogate.bind<int>("x", &x);

    surrogate.call_original_and_capture();
    REQUIRE(surrogate.get_captured_input<int>(0, 0) == 3);
}


