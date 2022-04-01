
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "optics.h"

// class Profunctor (p :: * -> * -> *) where
//   dimap :: (a' -> a) -> (b -> b') -> p a b -> p a' b'

template <template <typename, typename> typename P, typename A, typename B, typename AA, typename BB>
P<AA, BB> dimap (std::function<A(AA)>, std::function<BB(B)>, P<A,B>);


TEST_CASE("Demonstrate two-way binding of a primitive") {

    int x = 22;
    torch::Tensor t = torch::zeros({3,3});
    std::cout << t << std::endl;

    // Write out x into the tensor at index [1,1]
    PrimitiveAccessor<int> p(&x);
    p.fill_tensor(t, {1,1});
    std::cout << t << std::endl;

    // Modify the tensor at index [1,1]
    t[1][1] = 33;

    // Write back to primitive variable
    p.unfill_tensor(t, {1,1});
    std::cout << x << std::endl;
}

TEST_CASE("First test") {

    REQUIRE(true == false);

}