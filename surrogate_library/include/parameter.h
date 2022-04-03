
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
    virtual void stringify(std::ostream&, size_t sample_index) = 0;
    virtual ~Input() = default;
};

template <typename T>
struct InputT : public Input {
    optics::Optic<T>* accessor;
    std::vector<torch::Tensor> captures;
    Range<float> range;
    void stringify(std::ostream& os, size_t sample_index) override { os << captures[sample_index]; }
};

struct Output {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    virtual void stringify(std::ostream&, size_t sample_index) = 0;
    virtual ~Output() = default;
};

template <typename T>
struct OutputT : public Output {
    optics::Optic<T>* accessor;
    std::vector<torch::Tensor> captures;
    void stringify(std::ostream& os, size_t sample_index) override { os << captures[sample_index]; }
};


#endif //SURROGATE_TOOLKIT_PARAMETER_H
