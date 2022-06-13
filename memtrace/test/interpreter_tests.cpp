
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "interpreter.hpp"

using namespace phasm::memtrace;

TEST_CASE("Allocation tracker correctly handles a single entry") {
    Interpreter sut(1, {"main", "target", "f"});
    char x[4];
    sut.enter_fun(nullptr, 1, nullptr);
    sut.request_malloc(nullptr, 4);
    sut.receive_malloc(nullptr, &x);

    SECTION("Finds allocation given the exact address") {
        auto* result = sut.find_allocation_containing(x);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == x);
        REQUIRE(result->size == 4);
    }
    SECTION("Finds allocation given address contained within") {
        auto* result = sut.find_allocation_containing(x+1);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == x);
        REQUIRE(result->size == 4);
    }
    SECTION("Finds allocation given the upper-bound address") {
        auto* result = sut.find_allocation_containing(x+3);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == x);
        REQUIRE(result->size == 4);
    }
    SECTION("Doesn't find allocation given address too low") {
        auto* result = sut.find_allocation_containing(x-1);
        REQUIRE(result == nullptr);
    }
    SECTION("Doesn't find allocation given address too high") {
        auto *result = sut.find_allocation_containing(x + 4);
        REQUIRE(result == nullptr);
    }
}

TEST_CASE("Allocation tracker correctly handles multiple entries") {
    Interpreter sut(1, {"main", "target", "f"});
    int w[10];
    char x[4];
    double y;
    sut.enter_fun(nullptr, 1, nullptr);
    sut.request_malloc(nullptr, 10*sizeof(int));
    sut.receive_malloc(nullptr, w);
    sut.request_malloc(nullptr, 4*sizeof(char));
    sut.receive_malloc(nullptr, x);
    sut.request_malloc(nullptr, sizeof(double));
    sut.receive_malloc(nullptr, &y);

    SECTION("Finds allocation given the exact address") {
        auto* result = sut.find_allocation_containing(x);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == x);
        REQUIRE(result->size == 4);
    }
    SECTION("Finds allocation given address contained within") {
        auto* result = sut.find_allocation_containing(x+1);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == x);
        REQUIRE(result->size == 4);
    }
    SECTION("Finds allocation given the upper-bound address") {
        auto* result = sut.find_allocation_containing(x+3);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == x);
        REQUIRE(result->size == 4);
    }
    SECTION("Doesn't find allocation given address too low") {
        auto* result = sut.find_allocation_containing(x-1);
        REQUIRE((result == nullptr || result->addr != x));
    }
    SECTION("Doesn't find allocation given address too high") {
        auto *result = sut.find_allocation_containing(x+4);
        REQUIRE((result == nullptr || result->addr != x));
    }
}



TEST_CASE("Interpreter recognizes target fun reading external primitive") {
    int x;

    Interpreter sut(1, {"main", "target", "f"});
    sut.enter_fun(nullptr, 1, nullptr);
    sut.read_mem(nullptr, &x, 4, nullptr, nullptr);
    sut.exit_fun(nullptr);

    auto vars = sut.get_variables();
    REQUIRE(vars.size() == 1);
    REQUIRE(vars[0].sizes.size() == 1);
    REQUIRE(vars[0].sizes[0] == 4);
    REQUIRE(vars[0].is_input == true);
    REQUIRE(vars[0].is_output == false);
}

TEST_CASE("Interpreter recognizes allocation inside target function") {
    int x;
    Interpreter sut(1, {"main", "target", "f"});
    sut.enter_fun(nullptr, 1, nullptr);
    sut.request_malloc(nullptr, 4);
    sut.receive_malloc(nullptr, &x);
    sut.write_mem(nullptr, &x, 4, nullptr, nullptr);
    sut.exit_fun(nullptr);

    auto vars = sut.get_variables();
    REQUIRE(vars.size() == 1);
    REQUIRE(vars[0].sizes.size() == 1);
    REQUIRE(vars[0].sizes[0] == 4);
    REQUIRE(vars[0].is_input == false);
    REQUIRE(vars[0].is_output == true);
}