
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

TEST_CASE("Capturing inputs into a Parameters vector, single input") {
    CapturingFunction<double, double> cf(square);
    auto result = cf(7.0);
    REQUIRE(result == 49.0);
    REQUIRE(cf.m_parameters.size() == 2);
    REQUIRE(std::any_cast<double>(cf.m_parameters[0].data) == 7.0);
    REQUIRE(std::any_cast<double>(cf.m_parameters[1].data) == 49.0);
}


TEST_CASE("Capturing inputs into a Parameters vector, multiple inputs") {
    CapturingFunction<double, double, int, std::string> cf(f);
    auto result = cf(7.0, 3, "Nonsense");
    REQUIRE(result == 10.0);
    REQUIRE(cf.m_parameters.size() == 4);
    REQUIRE(std::any_cast<double>(cf.m_parameters[0].data) == 7.0);
    REQUIRE(std::any_cast<int>(cf.m_parameters[1].data) == 3);
    REQUIRE(std::any_cast<std::string>(cf.m_parameters[2].data) == "Nonsense");
    REQUIRE(std::any_cast<double>(cf.m_parameters[3].data) == 10.0);
}

TEST_CASE("Using a free function to avoid redundant template params ala std::make_tuple") {
    auto cf = make_capturing_function(f);
    auto result = cf(7.0, 3, "Nonsense");
    REQUIRE(result == 10.0);
}

TEST_CASE("Simplest possible reference-capturer") {
    double a = 22;
    double b = 44;
    double& c = std::ref(a);

    std::cout << "a before = " << a << std::endl;
    c = 33;
    std::cout << "a after = " << a << std::endl;

    c = std::ref(b);
    c = 55;
    std::cout << "b after = " << c << std::endl;

    std::tuple<double> t {4.0};
    double& d = std::get<0>(t);
    d = 33;
    REQUIRE(std::get<0>(t) == 33);

    rcf::ReferenceCapturingFunction<double, double> refcapfn(square);
    // Eventually use the capture mechanism from CapturingFunction instead of this nonsense
    auto p = static_cast<experiments::rcf::ParameterT<double>*>(refcapfn.m_parameters[0]);
    p->samples.push_back(6.0);
    REQUIRE(refcapfn() == 36.0);

}

TEST_CASE("Reference-capturer with different datatypes") {

    auto refcapfn = rcf::make_ref_capturing_function(f);
    // Eventually use the capture mechanism from CapturingFunction instead of this nonsense
    dynamic_cast<experiments::rcf::ParameterT<double>*>(refcapfn.m_parameters[0])->samples.push_back(6.0);
    dynamic_cast<experiments::rcf::ParameterT<int>*>(refcapfn.m_parameters[1])->samples.push_back(3);
    dynamic_cast<experiments::rcf::ParameterT<std::string>*>(refcapfn.m_parameters[2])->samples.push_back("Hello");
    REQUIRE(refcapfn() == 9.0);
}

TEST_CASE("Reference-capturer with uglier datatypes") {

    std::tuple<double> t {2.0};
    REQUIRE(std::apply(ref, t) == 4.0);

    // auto sut1 = rcf::make_ref_capturing_function(ref);
    // dynamic_cast<experiments::rcf::ParameterT<double>*>(sut1.m_parameters[0])->samples.push_back(6.0);
    // REQUIRE(sut1() == 36.0);

    // auto sut2 = rcf::make_ref_capturing_function(cref);
    // dynamic_cast<experiments::rcf::ParameterT<double>*>(sut1.m_parameters[0])->samples.push_back(6.0);
    // REQUIRE(sut2() == 36.0);

    // auto sut3 = rcf::make_ref_capturing_function(rref);
    // dynamic_cast<experiments::rcf::ParameterT<double>*>(sut3.m_parameters[0])->samples.push_back(6.0);
    // REQUIRE(sut3() == 36.0);

    auto sut3 = rcf::make_ref_capturing_function(csquare);
    dynamic_cast<experiments::rcf::ParameterT<double>*>(sut3.m_parameters[0])->samples.push_back(6.0);
    REQUIRE(sut3() == 36.0);


}


TEST_CASE("Something else") {
    // Basic idea: Instead of writing to a tuple which is hidden, write to the _original_ locations
    // Several different problems.
    // 1. Using a tuple encapsulated in the surrogate fn for the original fn's arguments doesn't work when args are refs or crefs or rrefs/
    // On the other hand, using the original locations doesn't work when
    //	1. the inputs are const
    //	2. the inputs are rvalues
    // Option 2 is more consistent with what we have to do for the non-argument inputs. The user can clean up the arguments
    // to get rid of consts or rvalues if necessary. Note this is only necessary for sampling, not for capturing!


    // Option 3: Best of all? Collect args as well as f at construction time, bind parameters to args immediately.
    // Capture works by calling f() OR f.capture() with no args
    // Sampling works by calling f.sample() which rewrites the bound inputs and outputs according to some algorithm
    // We keep the symmetry with

    // Consider doing something with sparse grids
}

