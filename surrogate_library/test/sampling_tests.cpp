
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "surrogate_builder.h"

int func(int a, int& b, const int& c, int&& d, int* e, const int* f) {
    return a + b + c + d + *e + *f;
}

TEST_CASE("Basic sample") {

    auto m = std::make_shared<Model>();
    m->input<int>("a");
    m->input<int>("b");
    m->output<int>("c");

    int a,b,c;
    Surrogate s([&](){c = a+b;}, m);
    s.bind_input("a", &a);
    s.bind_input("b", &b);
    s.bind_output("c", &c);

    a = 3; b = 7;
    s.set_sample_input(0, 17);
    s.set_sample_input(1, 19);

    REQUIRE(a == 3);
    REQUIRE(b == 7);

    s.call_original_with_sampled_inputs();

    REQUIRE(a == 17);
    REQUIRE(b == 19);
    REQUIRE(c == 36);
}

TEST_CASE("Basic sample using lambdas instead") {
    int a = 3;
    int b = 7;
    int c = 0;

    auto m = std::make_shared<Model>();
    m->input<int>("a");
    m->input<int>("b");
    m->output<int>("c");

    auto s = Surrogate([&](){c = a+b;}, m);
    s.bind_input<int>("a", &a);
    s.bind_input<int>("b", &b);
    s.bind_output<int>("c", &c);

    s.set_sample_input(0, 17);
    s.set_sample_input(1, 19);

    REQUIRE(a == 3);
    REQUIRE(b == 7);

    s.call_original_with_sampled_inputs();

    REQUIRE(a == 17);
    REQUIRE(b == 19);
    REQUIRE(c == 36);

}

TEST_CASE ("More extensive sample") {


    // We need to have actual lvalues where our samples can be written to
    int a = 19; int b = 20; int c = 21; int d = 22; int e = 23; int f = 24; int g = 0;

    auto m = std::make_shared<Model>();
    m->input<int>("a");
    m->input<int>("b");
    m->input<int>("c");
    m->input<int>("d");
    m->input<int>("e");
    m->input<int>("f");
    m->output<int>("g");
    auto sut = Surrogate([&](){ g = func(a,b,c,std::move(d),&e,&f);}, m);

    sut.bind_input<int>("a", &a);
    sut.bind_input<int>("b", &b);
    sut.bind_input<int>("c", &c);
    sut.bind_input<int>("d", &d);
    sut.bind_input<int>("e", &e);
    sut.bind_input<int>("f", &f);
    sut.bind_output<int>("g", &g);

    // Note that even though our original function demands rvalues,
    // pointers, const pointers, and const references, we can
    // still accommodate this, albeit slightly weirdly.

    // Now we set up a sample manually. Note that in real life,
    // the sample would be chosen by each parameter's Range objects
    // instead.
    sut.set_sample_input(0, 32);
    sut.set_sample_input(1, 33);
    sut.set_sample_input(2, 34);
    sut.set_sample_input(3, 35);
    sut.set_sample_input(4, 36);
    sut.set_sample_input(5, 37);

    REQUIRE(a == 19);
    REQUIRE(b == 20);
    REQUIRE(c == 21);
    REQUIRE(d == 22);
    REQUIRE(e == 23);
    REQUIRE(f == 24);

    sut.call_original_with_sampled_inputs();

    REQUIRE(a == 32);
    REQUIRE(b == 33);
    REQUIRE(c == 34);
    REQUIRE(d == 35);
    REQUIRE(e == 36);
    REQUIRE(f == 37);
    REQUIRE(g == 207);

    REQUIRE(sut.get_captured_input<int>(0, 0) == 32);
    REQUIRE(sut.get_captured_input<int>(0, 1) == 33);
    REQUIRE(sut.get_captured_input<int>(0, 2) == 34);
    REQUIRE(sut.get_captured_input<int>(0, 3) == 35);
    REQUIRE(sut.get_captured_input<int>(0, 4) == 36);
    REQUIRE(sut.get_captured_input<int>(0, 5) == 37);
    REQUIRE(sut.get_captured_output<int>(0, 0) == 207);

}