
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "experiments.h"
using namespace experiments;

double f(double x, int y, std::string) {
    return x + y;
}

double square(double x) {
    return x*x;
}

double csquare(const double x) {
    return x*x;
}

double ref(double& x) {
    return x*x;
}
double cref(const double& x) {
    return x*x;
}
double rref(double&& x) {
    return x*x;
}

TEST_CASE("Wrapping") {
    WrappedFunction<double, double> wf(square);
    REQUIRE(wf(3.0) == 9.0);

    double d = 3.0;
    WrappedFunction<double, double&> wf2(ref);
    REQUIRE(wf2(d) == 9.0);

    WrappedFunction<double, const double&> wf3(cref);
    REQUIRE(wf3(d) == 9.0);

    WrappedFunction<double, double&&> wf4(rref);
    REQUIRE(wf4(3.0) == 9.0);

    WrappedFunction<double, const double> wf5(csquare);
    REQUIRE(wf5(3.0) == 9.0);
}

TEST_CASE("Memoizing") {
    MemoizedFunction<double, double> mf(square);
    REQUIRE(mf(3.0) == 9.0);
    REQUIRE(mf.was_last_call_memoized == false);
    REQUIRE(mf(3.0) == 9.0);
    REQUIRE(mf.was_last_call_memoized == true);
    REQUIRE(mf(4.0) == 16.0);
    REQUIRE(mf.was_last_call_memoized == false);

    auto mf2 = MemoizedFunction<double, double, int, std::string>(f);
    REQUIRE(mf2(1, 2, "Hello") == 3.0);
    REQUIRE(mf2.was_last_call_memoized == false);
    REQUIRE(mf2(1, 2, "Hello") == 3.0);
    REQUIRE(mf2.was_last_call_memoized == true);

}

TEST_CASE("Currying") {
    CurriedFunction<double, double> cf(square, 2);
    REQUIRE(cf() == 4.0);

    CurriedFunction<double, double, int, std::string> cf2(f, 2, 3, "Test");
    REQUIRE(cf2() == 5.0);
}

template <typename T>
void registerParam(T t) {
    std::cout << "Registering " << t << std::endl;
}

TEST_CASE("Iterating over items in a tuple") {
    using TT = std::tuple<double, int, std::string>;
    TT t {3.3, 7, "Hello"};
    // For each argument, register a parameter
    std::apply([&](auto ... x){ (registerParam(x), ...); }, t);
}


