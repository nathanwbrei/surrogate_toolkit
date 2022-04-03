
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include <torch/torch.h>

TEST_CASE("Converting from C++ primitive types to dtypes") {

    // REQUIRE(true == false);
    double source[] = {22.2,33};
    auto ar = at::ArrayRef<double>(source, 2);
    auto t = torch::tensor(ar, torch::TensorOptions().dtype(torch::kF64));
    std::cout << t << std::endl;
    std::cout << t.dtype() << std::endl;
}


