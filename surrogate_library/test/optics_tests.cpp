
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>

// class Profunctor (p :: * -> * -> *) where
//   dimap :: (a' -> a) -> (b -> b') -> p a b -> p a' b'

template <template <typename, typename> typename P, typename A, typename B, typename AA, typename BB>
P<AA, BB> dimap (std::function<A(AA)>, std::function<BB(B)>, P<A,B>);





TEST_CASE("First test") {

    REQUIRE(true == false);

}