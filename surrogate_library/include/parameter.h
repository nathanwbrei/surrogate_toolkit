
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
    virtual void stringify(std::ostream&, size_t sample_index) = 0;
    virtual ~Input() = default;
};

template <typename T>
struct InputT : public Input {
    std::vector<T> captures;
    Range<T> range;
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
    std::vector<T> captures;
    void stringify(std::ostream& os, size_t sample_index) override { os << captures[sample_index]; }
};


#endif //SURROGATE_TOOLKIT_PARAMETER_H
