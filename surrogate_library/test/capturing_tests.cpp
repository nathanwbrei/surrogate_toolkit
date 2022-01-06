
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "surrogate_builder.h"

int mult(int x, int y) {
    return x * y;
}

TEST_CASE("Capture int(int,int)") {

    int x,y,z;
    auto surrogate = make_surrogate([&](){z = mult(x,y);});
    surrogate.input<int>("x", &x);
    surrogate.input<int>("y", &y);
    surrogate.output<int>("z", &z);

    x = 3; y = 5;
    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 5);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 15);
}

int mult_const(const int x, const int y) {
    return x * y;
}

TEST_CASE("Capture int(const int, const int)") {

    int x = 3,y = 5,z = 0;
    auto surrogate = make_surrogate([&](){z = mult(x,y);});
    surrogate.input<int>("x", &x);
    surrogate.input<int>("y", &y);
    surrogate.output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 5);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 15);
}

int mult_with_ref(int& x, int&& y) {
    return x * y;
}

TEST_CASE("Capture int(int&,int&&)") {

    int x = 3, y = 5, z = 0;
    auto surrogate = make_surrogate([&](){z = mult_with_ref(x,std::move(y));});
    surrogate.input<int>("x", &x);
    surrogate.input<int>("y", &y);
    surrogate.output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 5);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 15);
}

int mult_with_out_param(int& x, int y) {
    int z = x*y;
    x = 22;
    return z;
}

TEST_CASE("Capture int(int&,int) [input and output]") {

    int x=3, y=5, z=0;
    auto surrogate = make_surrogate([&](){z = mult_with_out_param(x,y);});
    surrogate.input_output<int>("x", &x);
    surrogate.input<int>("y", &y);
    surrogate.output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 15);
    REQUIRE(x == 22);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 5);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 22);
    REQUIRE(surrogate.getCapturedOutput<int>(0,1) == 15);
}

int g = 22;
int mult_with_global(int x) {
    return x * g;
}

TEST_CASE("Capture int(int) [with global]") {

    int x = 5, z = 0;
    auto surrogate = make_surrogate([&](){z = mult_with_global(x);});
    surrogate.input<int>("x", &x);
    surrogate.input<int>("g", &g);
    surrogate.output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 110);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 5);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 22);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 110);
}

int no_args(void) {
    return 0;
}

TEST_CASE("Capture int() [with no args]") {

    int z = 22;
    auto surrogate = make_surrogate([&](){z = no_args();});
    surrogate.output<int>("z", &z);

    surrogate.call_original_and_capture();
    REQUIRE(z == 0);

    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 0);
}

void return_void(int x) {
    x = x*x;
}

TEST_CASE("Capture void(int)") {

    int x = 3;
    auto surrogate = make_surrogate([&](){return_void(x);});
    surrogate.input<int>("x", &x);

    surrogate.call_original_and_capture();
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
}

template <typename R, typename... A>
struct Surr {
    R test(A&&...) {
        R r;
        return r;
    }
};

template <typename... A>
struct Surr<void, A...> {
    void test(A&&...){
    }
};
TEST_CASE("Wild and crazy template stuff") {
    Surr<void, int> s;
    s.test(3);
    // Idea: we can handle void return types by partial template
    // specialization just like above. This is going to be really
    // ugly because we will be duplicating almost all of the code,
    // but it should definitely work.
}

/*
TEST_CASE("Ugliness when passing in lvalues") {

    auto sut = make_surrogate(mult);
    sut.input<int, 0>("a");
    sut.input<int, 1>("b");
    sut.returns<int>("c");

    int a = 3;
    int b = 7;

    int result = sut.call_original_and_capture(std::forward<int>(a), std::forward<int>(b));
    REQUIRE(result == 21);

}
*/



