
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_VARIABLE_H
#define SURROGATE_TOOLKIT_MODEL_VARIABLE_H

#include "range.h"
#include "optics.h"
#include "any_ptr.hpp"
#include "tensor.hpp"


namespace phasm {

enum Direction {IN, OUT, INOUT};

struct ModelVariable {
    std::string name;
    bool is_input = false;
    bool is_output = false;
    OpticBase *accessor = nullptr;
    std::vector<tensor> training_inputs;
    std::vector<tensor> training_outputs;
    tensor inference_input;
    tensor inference_output;
    Range range;

    ModelVariable() = default;
    ~ModelVariable() {
        delete accessor;
    }

    std::vector<int64_t> shape() const {
        if (accessor == nullptr) {
            std::ostringstream oss;
            oss << "ModelVariable '" << name << "' doesn't have an accessor";
            throw std::runtime_error(oss.str());
        }
        return accessor->shape();
    }

    void captureTrainingInput(const phasm::any_ptr &binding) {
        tensor data = accessor->unsafe_to(binding);
        training_inputs.push_back(data);
    }

    void captureTrainingOutput(const phasm::any_ptr &binding) {
        tensor data = accessor->unsafe_to(binding);
        training_outputs.push_back(data);
    }

    void captureInferenceInput(const phasm::any_ptr &binding) {
        inference_input = accessor->unsafe_to(binding);
    }

    void publishInferenceOutput(const phasm::any_ptr &binding) const {
        accessor->unsafe_from(inference_output, binding);
    }
};

} // namespace phasm
#endif //SURROGATE_TOOLKIT_MODEL_VARIABLE_H
