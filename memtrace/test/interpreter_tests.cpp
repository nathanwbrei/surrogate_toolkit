
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "interpreter.hpp"

using namespace phasm::memtrace;

TEST_CASE("Allocation tracker correctly handles a single entry") {
    const uintptr_t TARGET_FUN_IP = 0;
    const uintptr_t X = 256;  // Pretend x :: char[4]

    Interpreter sut(TARGET_FUN_IP);
    sut.enter_fun(TARGET_FUN_IP, 0);
    sut.request_malloc(TARGET_FUN_IP+1, 4);
    sut.receive_malloc(TARGET_FUN_IP+2, X);

    SECTION("Finds allocation given the exact address") {
        auto* result = sut.find_allocation_containing(X);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == X);
        REQUIRE(result->size == 4);
    }
    SECTION("Finds allocation given address contained within") {
        auto* result = sut.find_allocation_containing(X+1);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == X);
        REQUIRE(result->size == 4);
    }
    SECTION("Finds allocation given the upper-bound address") {
        auto* result = sut.find_allocation_containing(X+3);
        REQUIRE(result != nullptr);
        REQUIRE(result->addr == X);
        REQUIRE(result->size == 4);
    }
    SECTION("Doesn't find allocation given address too low") {
        auto* result = sut.find_allocation_containing(X-1);
        REQUIRE(result == nullptr);
    }
    SECTION("Doesn't find allocation given address too high") {
        auto *result = sut.find_allocation_containing(X+4);
        REQUIRE(result == nullptr);
    }
}

TEST_CASE("Allocation tracker correctly handles multiple entries") {

    const uintptr_t TARGET_IP = 512;
    Interpreter sut(TARGET_IP);
    uintptr_t w = 200; // Pretend typeof(w) == int[10]
    uintptr_t x = w + 10*sizeof(int); // Pretend typeof(x) == char[4]
    uintptr_t y = x + 4*sizeof(char); // Pretend typeof(x) == double

    sut.enter_fun(TARGET_IP, 0);
    sut.request_malloc(TARGET_IP+1, 10*sizeof(int));
    sut.receive_malloc(TARGET_IP+2, w);
    sut.request_malloc(TARGET_IP+3, 4*sizeof(char));
    sut.receive_malloc(TARGET_IP+4, x);
    sut.request_malloc(TARGET_IP+5, sizeof(double));
    sut.receive_malloc(TARGET_IP+6, y);

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
    uintptr_t x = 234;
    uintptr_t TARGET_IP = 0;

    Interpreter sut(TARGET_IP);
    sut.enter_fun(TARGET_IP, 0);
    sut.read_mem(TARGET_IP+1, x, 4, 0, 0);
    sut.exit_fun(TARGET_IP+2);

    auto vars = sut.get_variables();
    REQUIRE(vars.size() == 1);
    REQUIRE(vars[0].sizes.size() == 1);
    REQUIRE(vars[0].sizes[0] == 4);
    REQUIRE(vars[0].is_input == true);
    REQUIRE(vars[0].is_output == false);
}

TEST_CASE("Interpreter recognizes allocation inside target function") {
    uintptr_t TARGET_IP = 77;
    int x;
    Interpreter sut(TARGET_IP);
    sut.enter_fun(TARGET_IP, 0);
    sut.request_malloc(TARGET_IP+1, 4);
    sut.receive_malloc(TARGET_IP+2, reinterpret_cast<uintptr_t>(&x));
    sut.write_mem(3, reinterpret_cast<uintptr_t>(&x), 4, 0, 0);
    sut.exit_fun(4);

    auto vars = sut.get_variables();
    REQUIRE(vars.size() == 1);
    REQUIRE(vars[0].sizes.size() == 1);
    REQUIRE(vars[0].sizes[0] == 4);
    REQUIRE(vars[0].is_input == false);
    REQUIRE(vars[0].is_output == true);
}