

#include <tensor.hpp>
#include <catch.hpp>
#include <omnitensor.hpp>

namespace phasm {

TEST_CASE("OmnitensorBasic") {

    SECTION("1x1") {
        phasm::omnitensor t = phasm::omnitensor(phasm::DType::F32,{1},1);
        REQUIRE(t.length() == 1);
        REQUIRE(t.offset(index(std::vector<int>{0})) == 0);
        REQUIRE(t.offset(index(std::vector<int>{0})) < t.length());
    }
    SECTION("3x5") {
        phasm::omnitensor t = phasm::omnitensor(phasm::DType::F32,{3,5},1);
        REQUIRE(t.length() == 15);
        REQUIRE(t.offset(index(std::vector<int>{1,1})) == 6);
        REQUIRE(t.length(index(std::vector<int>{1,1})) == 1);

        REQUIRE(t.offset(index(std::vector<int>{1})) == 5);
        REQUIRE(t.length(index(std::vector<int>{1})) == 5);

        REQUIRE(t.offset(index(std::vector<int>{})) == 0);
        REQUIRE(t.length(index(std::vector<int>{})) == 15);
    }

}

} // namespace phasm

