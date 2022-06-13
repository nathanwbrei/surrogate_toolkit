
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "tensor.hpp"

namespace phasm {

tensor stack(std::vector<tensor>& tensors) {
    std::vector<torch::Tensor> torch_tensors;
    torch_tensors.reserve(tensors.size());
    for (auto t : tensors) {
        torch_tensors.push_back(t.get_underlying());
    }
    return phasm::tensor(torch::stack(torch_tensors));
}

std::vector<tensor> unstack(tensor& t) {
    std::vector<phasm::tensor> phasm_unstacked;
    phasm_unstacked.reserve(t.get_underlying().size(0));
    std::vector<torch::Tensor> torch_unstacked = torch::unbind(t.get_underlying(), 0);
    for (auto tu : torch_unstacked) {
        phasm_unstacked.emplace_back(tu);
    }
    return phasm_unstacked;
}

tensor flatten(tensor& t) {
    auto flattened = t.get_underlying().flatten(0, -1);
    return phasm::tensor(flattened);
}

} // namespace phasm

