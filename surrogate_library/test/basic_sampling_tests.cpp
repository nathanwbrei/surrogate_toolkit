
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "surrogate_builder.h"

int plus(int a, int b) {
    return a + b;
}

int func(int a, int& b, const int& c, int&& d, int* e, const int* f) {
    return a + b + c + d + *e + *f;
}

TEST_CASE("Basic sample") {

    auto sut = make_surrogate(plus);
    sut.input<int, 0>("a");
    sut.input<int, 1>("b");
    sut.returns<int>("c");

    int a = 3;
    int b = 7;

    sut.setSampleInput(0, 17);
    sut.setSampleInput(1, 19);

    REQUIRE(a == 3);
    REQUIRE(b == 7);

    int result = sut.call_original_with_sampled_inputs(std::forward<int>(a), std::forward<int>(b));

    // REQUIRE(a == 17);
    // REQUIRE(b == 19);
    // REQUIRE(result == 36);
}

TEST_CASE("Basic sample using lambdas instead") {
    int a = 3;
    int b = 7;
    // auto sut = make_surrogate([&](){return plus(a,b);});

    auto sut = Surrogate<int>([&](){return plus(a,b);});
    sut.input<int>("a", [&](){return &a;});
    sut.input<int>("b", [&](){return &b;});
    sut.returns<int>("c");

    sut.setSampleInput(0, 17);
    sut.setSampleInput(1, 19);

    REQUIRE(a == 3);
    REQUIRE(b == 7);

    int result = sut.call_original_with_sampled_inputs();

    REQUIRE(a == 17);
    REQUIRE(b == 19);
    REQUIRE(result == 36);

}

TEST_CASE ("More extensive sample") {

    auto sut = make_surrogate(func);

    // We need to have actual lvalues where our samples can be written to
    int a = 19;
    int b = 20;
    int c = 21;
    int d = 22;
    int e = 23;
    int f = 24;

    sut.input<int, 0>("a");
    sut.input<int, 1>("b");

    // Note that even though our original function demands rvalues,
    // pointers, const pointers, and const references, we can
    // still accommodate this, albeit slightly weirdly.

    sut.input<int>("c", [&](){return &c;});
    sut.input<int>("d", [&](){return &d;});
    sut.input<int>("e", [&](){return &e;});
    sut.input<int>("f", [&](){return &f;});
    sut.returns<int>("g");

    // Now we set up a sample manually. Note that in real life,
    // the sample would be chosen by each parameter's Range objects
    // instead.
    sut.setSampleInput(0, 32);
    sut.setSampleInput(1, 33);
    sut.setSampleInput(2, 34);
    sut.setSampleInput(3, 35);
    sut.setSampleInput(4, 36);
    sut.setSampleInput(5, 37);

    REQUIRE(a == 19);
    REQUIRE(b == 20);
    REQUIRE(c == 21);
    REQUIRE(d == 22);
    REQUIRE(e == 23);
    REQUIRE(f == 24);


    int result = sut.call_original_with_sampled_inputs(std::forward<int>(a), b, c, std::forward<int>(d), &e, &f);

    // REQUIRE(a == 32);
    // REQUIRE(b == 33);
    REQUIRE(c == 34);
    REQUIRE(d == 35);
    REQUIRE(e == 36);
    REQUIRE(f == 37);
    // REQUIRE(result == 207);

    // REQUIRE(sut.getCapturedInput<int>(0,0) == 32);
    // REQUIRE(sut.getCapturedInput<int>(0,1) == 33);
    REQUIRE(sut.getCapturedInput<int>(0,2) == 34);
    REQUIRE(sut.getCapturedInput<int>(0,3) == 35);
    REQUIRE(sut.getCapturedInput<int>(0,4) == 36);
    REQUIRE(sut.getCapturedInput<int>(0,5) == 37);
    // REQUIRE(sut.getCapturedOutput<int>(0,0) == 207);

}