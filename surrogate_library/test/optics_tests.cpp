
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "optics.h"

// class Profunctor (p :: * -> * -> *) where
//   dimap :: (a' -> a) -> (b -> b') -> p a b -> p a' b'

template <template <typename, typename> typename P, typename A, typename B, typename AA, typename BB>
P<AA, BB> dimap (std::function<A(AA)>, std::function<BB(B)>, P<A,B>);


TEST_CASE("Demonstrate two-way binding of a primitive") {

    int32_t x = 22;

    // Write out x into the tensor at index [1,1]
    optics::Primitive<int> p;
    auto t = p.to(&x);

    std::cout << t.dtype();
    int32_t* tp = t[0].data_ptr<int>();
    REQUIRE(*tp == 22);

    // Modify the tensor
    *tp = 33;

    // Write back to primitive variable
    p.from(t, &x);
    REQUIRE(x == 33);
}

TEST_CASE("Pytorch simplest possible tensor") {
    float x = 22.2;
    auto t = torch::tensor(at::ArrayRef(&x, 1), torch::TensorOptions().dtype(torch::kFloat32));
    REQUIRE(t.size(0) == 1);
    REQUIRE(t.dim() == 1);
    auto tt = torch::tensor({22.2});
    float y = *(t[0].data_ptr<float>());
    REQUIRE(x == y);
}

TEST_CASE("PyTorch tensor operations") {

   /*
    float arr[9] = {1.2,2,3,4,5,6,7,8,9};
    std::vector<size_t> shape {3,3};
    std::vector<size_t> strides {1,3};
    auto t = torch::tensor({arr, 9});
    std::cout << t << std::endl;
    // auto t = torch::from_blob(arr);// , shape, strides).clone();
    */
}

struct MyStruct {
    float x;
    float y;
};

TEST_CASE("Composition of a Field lens with a Primitive") {
    MyStruct s { 49.0, 7.6};

    auto primitive_lens = optics::Primitive<float>();
    auto getY = [](MyStruct* s){return &(s->y);};
    auto field_lens = optics::Field<MyStruct, float>(&primitive_lens, getY);
    // auto field_lens = optics::Field(primitive_lens, getY);

    // Obviously we need to get template type deduction working... maybe just use a newer compiler?
    // auto field_lens = optics::make_field_lens(primitive_lens, [](MyStruct* s){return &(s->y);});

    // Should extract y and stick it in a tensor
    auto t = field_lens.to(&s);
    float* tp = t[0].data_ptr<float>();
    REQUIRE(*tp == (float) 7.6);

}

struct OtherStruct {
    int w;
    MyStruct* s;
};

TEST_CASE("Composition of two structs") {
    MyStruct ms {1,2};
    OtherStruct os {3, &ms};

    auto primitive_lens = optics::Primitive<float>();
    auto getY = [](MyStruct* s){return &(s->y);};
    auto inner_lens = optics::Field<MyStruct, float>(&primitive_lens, getY);
    auto getMs = [](OtherStruct* os){return os->s;};
    auto outer_lens = optics::Field<OtherStruct, MyStruct>(&inner_lens, getMs);

    auto t = outer_lens.to(&os);
    float* tp = t[0].data_ptr<float>();
    REQUIRE(*tp == (float) 2.0);
}


TEST_CASE("Array of structs") {
    MyStruct aos[5] = {{1,2},{5,6},{10,11},{15,16},{20,21}};
    auto primitive_iso = optics::Primitive<float>();
    auto inner_lens = optics::Field<MyStruct, float>(&primitive_iso, [](MyStruct* s){return &(s->y);});
    auto array_traversal = optics::Array<MyStruct>(&inner_lens, 5);

    auto t = array_traversal.to(aos);
    std::cout << t << std::endl;
    REQUIRE(t.size(0) == 5);
    REQUIRE(t.size(1) == 1);
}


TEST_CASE("1-D Array of Primitive produces same Tensor as PrimitiveArray") {
    int xs[] = {1,2,3,4,5};
    auto primitive_iso = optics::Primitive<int>();
    auto array_trav = optics::Array<int>(&primitive_iso, 5);
    auto primitive_array_iso = optics::PrimitiveArray<int>({5,1});  // Note we can also specify shape as {5}

    auto t1 = array_trav.to(xs);
    auto t2  = primitive_array_iso.to(xs);

    REQUIRE(*torch::all(t1 == t2).data_ptr<bool>() == true);
}


