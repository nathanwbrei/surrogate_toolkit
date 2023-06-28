
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include <optics.h>
#include <torch/torch.h>

using namespace phasm;
namespace phasm::tests::pytorch_tests {

TEST_CASE("Pytorch simplest possible tensor") {
    float x = 22.2;
    auto t = torch::tensor(at::ArrayRef(&x, 1), torch::TensorOptions().dtype(torch::kFloat32));
    REQUIRE(t.size(0) == 1);
    REQUIRE(t.dim() == 1);
    auto tt = torch::tensor({22.2});
    float y = *(t[0].data_ptr<float>());
    REQUIRE(x == y);
}

/*
TEST_CASE("PyTorch tensor operations") {

 float arr[9] = {1.2,2,3,4,5,6,7,8,9};
 std::vector<size_t> shape {3,3};
 std::vector<size_t> strides {1,3};
 auto t = torch::tensor({arr, 9});
 std::cout << t << std::endl;
 // auto t = torch::from_blob(arr);// , shape, strides).clone();
}
 */

TEST_CASE("Getting a multidimensional array into a Torch tensor") {

    double mat[2][3] = {{1,2,3},{4,5,6}};
    auto matt = torch::tensor(at::ArrayRef<double>(mat[0],6), torch::dtype<double>());
    matt = matt.reshape({2,3});
    std::cout << matt << std::endl;
}

TEST_CASE("Getting a multidimensional array out of a Torch tensor") {

    double mat[2][3] = {{1,2,3},{4,5,6}};
    auto matt = torch::tensor(at::ArrayRef<double>(mat[0],6), torch::dtype<double>());
    matt = matt.reshape({2,3});

    matt += 7;

    /// Shape has to be exact. This lets us not have to worry about our array being contiguous,
    /// but also prevents us from using accessors when we don't know the shape at compile time.
    auto matt_a = matt.accessor<double, 2>();
    for (size_t row=0; row<2; ++row) {
        for (size_t col=0; col<3; ++col) {
            mat[row][col] = matt_a[row][col];
        }
    }

    std::cout << mat[0][0] << " " << mat[0][1] << " " << mat[0][2] << std::endl;
    std::cout << mat[1][0] << " " << mat[1][1] << " " << mat[1][2] << std::endl;
}


TEST_CASE("Getting a multidimensional array out of a Torch tensor with arbitrary shape") {

    double mat[2][3] = {{1,2,3},{4,5,6}};
    auto matt = torch::tensor(at::ArrayRef<double>(mat[0],6), torch::dtype<double>());
    matt = matt.reshape({2,3});

    matt += 7;

    double* matt_ptr = matt.data_ptr<double>();
    for (int idx = 0; idx<6; ++idx) {
        mat[0][idx] = matt_ptr[idx];
    }

    std::cout << mat[0][0] << " " << mat[0][1] << " " << mat[0][2] << std::endl;
    std::cout << mat[1][0] << " " << mat[1][1] << " " << mat[1][2] << std::endl;
}



}
