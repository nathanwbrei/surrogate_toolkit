
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "surrogate_builder.h"
#include "sampler.h"

using namespace phasm;

namespace phasm::tests::sampling_tests {

template <typename T>
phasm::tensor scalar_to_tensor(T scalar) {
    return tensor(&scalar, 1);
}


// TODO: Re-work this
#if 0
TEST_CASE("Basic GridSampler") {

    int x = 100;
    auto v = SurrogateBuilder().local_primitive<int>("x", INOUT).get_model_vars()[0];
    v->range.lower_bound_inclusive = scalar_to_tensor(3);
    v->range.upper_bound_inclusive = scalar_to_tensor(5);
    GridSampler s(v, 1);

    bool result;
    result = s.next();
    REQUIRE(result == true);
    REQUIRE(x == 3);

    result = s.next();
    REQUIRE(result == true);
    REQUIRE(x == 4);

    result = s.next();
    REQUIRE(result == false);
    REQUIRE(x == 5);

    result = s.next();
    REQUIRE(result == true); // Wraps around
    REQUIRE(x == 3);
}
#endif

// TODO: Re-work this.
#if 0
TEST_CASE("Basic FiniteSetSampler") {
    auto v = SurrogateBuilder().local_primitive<int>("x", INOUT).get_model_vars()[0];
    v->range.items.insert(scalar_to_tensor(7));
    v->range.items.insert(scalar_to_tensor(8));
    v->range.items.insert(scalar_to_tensor(9));
    v->range.rangeType = RangeType::FiniteSet;
    FiniteSetSampler s(v);

    int x = 100;

    bool result;
    result = s.next();
    REQUIRE(result == true);
    v->publishInferenceOutput(&x);
    REQUIRE(x == 7);

    result = s.next();
    REQUIRE(result == true);
    REQUIRE(x == 8);

    result = s.next();
    REQUIRE(result == false);
    REQUIRE(x == 9);

    result = s.next();
    REQUIRE(result == true); // Wraps around
    REQUIRE(x == 7);
}
#endif

} // namespace phasm::test::sampling_tests
