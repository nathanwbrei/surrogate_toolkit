
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_VARIABLE_H
#define SURROGATE_TOOLKIT_MODEL_VARIABLE_H

#include "range.h"
#include "optics.h"
#include "any_ptr.hpp"
#include <torch/torch.h>



struct ModelVariable {
    std::string name;
    optics::OpticBase* accessor = nullptr;
    phasm::any_ptr global;
    std::vector<torch::Tensor> training_captures;
    torch::Tensor inference_capture;
    Range<float> range;

    std::vector<int64_t> shape() {
        if (accessor == nullptr) {
            std::ostringstream oss;
            oss << "ModelVariable '" << name << "' doesn't have an accessor";
            throw std::runtime_error(oss.str());
        }
        return accessor->shape();
    }

    void capture_training_data(phasm::any_ptr binding) {
        torch::Tensor data = accessor->unsafe_to(binding);
        training_captures.push_back(data);
    }
    void get_inference_data(phasm::any_ptr binding) {
        inference_capture = accessor->unsafe_to(binding);
    }
    void put_inference_data(phasm::any_ptr binding) {
        accessor->unsafe_from(inference_capture, binding);
    }
};


#endif //SURROGATE_TOOLKIT_MODEL_VARIABLE_H
