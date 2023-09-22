
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include <iostream>
#include "optics.h"

using namespace phasm;
namespace phasm::test::optics_tests {

// class Profunctor (p :: * -> * -> *) where
//   dimap :: (a' -> a) -> (b -> b') -> p a b -> p a' b'

template<template<typename, typename> typename P, typename A, typename B, typename AA, typename BB>
P<AA, BB> dimap(std::function<A(AA)>, std::function<BB(B)>, P<A, B>);


TEST_CASE("Demonstrate two-way binding of a primitive") {

    int32_t x = 22;

    // Write out x into the tensor at index [1,1]
    TensorIso<int> p;
    auto t = p.to(&x);

    print_dtype(std::cout, t.get_dtype());
    int32_t *tp = t.get_data<int>();
    REQUIRE(*tp == 22);

    // Modify the tensor
    *tp = 33;

    // Write back to primitive variable
    p.from(t, &x);
    REQUIRE(x == 33);
}

TEST_CASE("Basic use of TensorIso") {
    double mat[2][3] = {{1,2,3},{4,5,6}};
    auto p = TensorIso<double>({2, 3});
    auto t = p.to(mat[0]);

    auto firstitem = *(t.get_data<double>());
    REQUIRE(firstitem == 1.0);
    REQUIRE(t.get_shape()[0] == 2);
    REQUIRE(t.get_shape()[1] == 3);
}

struct MyStruct {
    float x;
    float y;
};

TEST_CASE("Composition of a Field lens with a TensorIso") {
    MyStruct s{49.0, 7.6};

    auto primitive_lens = TensorIso<float>();
    auto getY = [](MyStruct *s) { return &(s->y); };
    auto field_lens = Lens<MyStruct, float>(&primitive_lens, getY);
    // auto field_lens = Field(primitive_lens, getY);

    // Obviously we need to get template type deduction working... maybe just use a newer compiler?
    // auto field_lens = make_field_lens(primitive_lens, [](MyStruct* s){return &(s->y);});

    // Should extract y and stick it in a tensor
    auto t = field_lens.to(&s);
    float *tp = t.get_data<float>();
    REQUIRE(*tp == (float) 7.6);

}


TEST_CASE("Composition of a RefLens with a TensorIso") {
    MyStruct s{49.0, 7.6};

    auto primitive_lens = TensorIso<float>();
    auto field_lens = RefLens<MyStruct, float>(&primitive_lens, &MyStruct::y);
    auto t = field_lens.to(&s);
    float *tp = t.get_data<float>();
    REQUIRE(*tp == (float)7.6);
    *tp = 99;
    field_lens.from(t, &s);
    REQUIRE(s.y == 99);
}

struct OtherStruct {
    int w;
    MyStruct *s;
};

TEST_CASE("Composition of two structs") {
    MyStruct ms{1, 2};
    OtherStruct os{3, &ms};

    auto primitive_lens = TensorIso<float>();
    auto getY = [](MyStruct *s) { return &(s->y); };
    auto inner_lens = Lens<MyStruct, float>(&primitive_lens, getY);
    auto getMs = [](OtherStruct *os) { return os->s; };
    auto outer_lens = Lens<OtherStruct, MyStruct>(&inner_lens, getMs);

    auto t = outer_lens.to(&os);
    float *tp = t.get_data<float>();
    REQUIRE(*tp == (float) 2.0);
}


TEST_CASE("Array of structs") {
    MyStruct aos[5] = {{1,  2},
                       {5,  6},
                       {10, 11},
                       {15, 16},
                       {20, 21}};
    auto primitive_iso = TensorIso<float>();
    auto inner_lens = Lens<MyStruct, float>(&primitive_iso, [](MyStruct *s) { return &(s->y); });
    auto array_traversal = ArrayTraversal<MyStruct>(&inner_lens, 5);

    auto t = array_traversal.to(aos);
    t.print(std::cout);
    REQUIRE(t.get_shape()[0] == 5);
}


TEST_CASE("1-D Array of TensorIso produces same Tensor as TensorIsoArray") {
    int xs[] = {1, 2, 3, 4, 5};
    auto primitive_iso = TensorIso<int>();
    auto array_trav = ArrayTraversal<int>(&primitive_iso, 5);
    auto primitive_array_iso = TensorIso<int>({5});  // Note we can also specify shape as {5}

    auto t1 = array_trav.to(xs);
    auto t2 = primitive_array_iso.to(xs);

    bool is_equal = true;
    int* t1_data = t1.get_data<int>();
    int* t2_data = t2.get_data<int>();
    for (size_t i=0; i<t1.get_length(); ++i) {
        is_equal &= (t1_data[i] == t2_data[i]);
    }

    REQUIRE(is_equal == true);
}

TEST_CASE("Iterate over std::vector") {
    std::vector<MyStruct> aos = {{1,  2},
                                 {5,  6},
                                 {10, 11},
                                 {15, 16},
                                 {20, 21}};
    auto primitive_iso = TensorIso<float>();
    auto inner_lens = Lens<MyStruct, float>(&primitive_iso, [](MyStruct *s) { return &(s->y); });
    using IterT = STLIterator<std::vector<MyStruct>, MyStruct>;
    auto vector_traversal = Traversal<std::vector<MyStruct>, MyStruct, IterT>(&inner_lens, 5);

    auto t = vector_traversal.to(&aos);
    t.print(std::cout);
    REQUIRE(t.get_shape()[0] == 5);
}

struct MyOrderableStruct {
    float x;
    float y;

    bool operator<(const MyOrderableStruct &other) {
        if (x < other.x) return true;
        if (x == other.x && y < other.y) return true;
        return false;
    }
};


/*
TEST_CASE("Iterate over std::set just like above") {
    std::set<MyOrderableStruct> aos = {{1,2},{5,6},{10,11},{15,16},{20,21}};
    auto primitive_iso = TensorIso<float>();
    auto inner_lens = Field<MyOrderableStruct, float>(&primitive_iso, [](MyOrderableStruct* s){return &(s->y);});
    using IterT = STLIterator<std::set<MyOrderableStruct>, MyOrderableStruct>;
    auto vector_traversal = Traversal<std::set<MyOrderableStruct>, MyOrderableStruct, IterT>(&inner_lens, 5);

    auto t = vector_traversal.to(&aos);
    std::cout << t << std::endl;
    REQUIRE(t.size(0) == 5);
    REQUIRE(t.size(1) == 1);
}
*/
// Write-back into containers such as sets and maps is a no-go. Their iterators are always const because otherwise
// we could break the container. I think the correct thing to do in this case is to clear the container and re-enter
// all of the values.


struct Tree {
    int data = 0;
    Tree *left = nullptr;
    Tree *right = nullptr;

    Tree(int x) : data(x) {};

    void Insert(int new_data) {
        if (new_data < data) {
            if (left == nullptr) {
                left = new Tree(new_data);
            } else {
                left->Insert(new_data);
            }
        } else {
            if (right == nullptr) {
                right = new Tree(new_data);
            } else {
                right->Insert(new_data);
            }
        }
    }
};
} // namespace phasm::test::optics_tests} // namespace phasm::test::optics_tests
