
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_PARAMETER_H
#define SURROGATE_TOOLKIT_PARAMETER_H

#include "range.h"
#include "optics.h"
#include <torch/torch.h>

enum class ParameterCategory {
    Continuous, Discrete, Categorical, Text
};

struct Input {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    std::vector<torch::Tensor> captures;
    virtual std::vector<int64_t> shape() = 0;
    virtual ~Input() = default;
};

template <typename T>
struct InputT : public Input {
    optics::Optic<T>* accessor;
    Range<float> range;
    std::vector<int64_t> shape() override {
        if (accessor == nullptr) { throw std::runtime_error("InputT needs an accessor"); }
        return accessor->shape();
    }
};

struct Output {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    std::vector<torch::Tensor> captures;
    virtual std::vector<int64_t> shape() = 0;
    virtual ~Output() = default;
};

template <typename T>
struct OutputT : public Output {
    optics::Optic<T>* accessor;
    std::vector<int64_t> shape() override {
        if (accessor == nullptr) { throw std::runtime_error("InputT needs an accessor"); }
        return accessor->shape();
    }
};


#endif //SURROGATE_TOOLKIT_PARAMETER_H
