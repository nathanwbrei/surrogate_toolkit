
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_PARAMETER_H
#define SURROGATE_TOOLKIT_PARAMETER_H

#include "range.h"

enum class ParameterCategory {
    Continuous, Discrete, Categorical, Text
};

struct Input {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    virtual ~Input() = default;
};

template <typename T>
struct InputT : public Input {
    std::vector<T> captures;
    T sample; // TODO: Does this live on param or param binding?
    Range<T> range;
};

struct Output {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    virtual ~Output() = default;
};

template <typename T>
struct OutputT : public Output {
    std::vector<T> captures;
};


#endif //SURROGATE_TOOLKIT_PARAMETER_H
