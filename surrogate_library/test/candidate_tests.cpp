
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "candidate.h"

int mult(int x, int y) {
    return x * y;
}

TEST_CASE("Capture int(int,int)") {

    auto surrogate = make_surrogate(mult);
    surrogate.input<int,0>("x");
    surrogate.input<int,1>("y");
    surrogate.returns<int>("z");

    REQUIRE(surrogate.call_original_and_capture_sample(3, 5) == 15);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 5);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 15);
}

int mult_with_ref(int& x, int&& y) {
    return x * y;
}

TEST_CASE("Capture int(int&,int&&)") {

    auto surrogate = make_surrogate(mult_with_ref);
    surrogate.input<int,0>("x");
    surrogate.input<int,1>("y");
    surrogate.returns<int>("z");
    int x = 3;
    REQUIRE(surrogate.call_original_and_capture_sample(x, 5) == 15);
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

    auto surrogate = make_surrogate(mult_with_out_param);
    surrogate.input_output<int,0>("x");
    surrogate.input<int,1>("y");
    surrogate.returns<int>("z");

    int x = 3;
    REQUIRE(surrogate.call_original_and_capture_sample(x, 5) == 15);
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

    auto surrogate = make_surrogate(mult_with_global);
    surrogate.input<int,0>("x");
    surrogate.input<int>("g", [](){return &g;});
    surrogate.returns<int>("z");

    REQUIRE(surrogate.call_original_and_capture_sample(5) == 110);
    REQUIRE(surrogate.getCapturedInput<int>(0,0) == 5);
    REQUIRE(surrogate.getCapturedInput<int>(0,1) == 22);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 110);
}

int no_args(void) {
    return 0;
}

TEST_CASE("Capture int() [with no args]") {

    auto surrogate = make_surrogate(no_args);
    surrogate.returns<int>("z");

    REQUIRE(surrogate.call_original_and_capture_sample() == 0);
    REQUIRE(surrogate.getCapturedOutput<int>(0,0) == 0);
}

void return_void(int x) {
    x = x*x;
}

TEST_CASE("Capture void(int)") {

    // auto surrogate = make_surrogate(return_void);
    // surrogate.input<int,0>("x");

    // surrogate.call_original_and_capture_sample(3);
    // REQUIRE(surrogate.getCapturedInput<int>(0,0) == 3);
}

template <typename R, typename... A>
struct Surr {
    R test(A&&... a) {
        R r;
        return r;
    }
};

template <typename... A>
struct Surr<void, A...> {
    void test(A&&... a){
        return;
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

