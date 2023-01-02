
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#pragma once

#include <torch/torch.h>
#include <tensor.hpp>

namespace phasm {

torch::Tensor to_torch_tensor(const phasm::tensor& t);

phasm::tensor to_phasm_tensor(const torch::Tensor& t);

torch::Tensor flatten_and_join(std::vector<torch::Tensor> inputs);

std::vector<torch::Tensor> split_and_unflatten_outputs(torch::Tensor output,
                                                       const std::vector<int64_t>& lengths,
                                                       const std::vector<std::vector<int64_t>>& shapes);

phasm::DType to_phasm_dtype(torch::Dtype t);

}




