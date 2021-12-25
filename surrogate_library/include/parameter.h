
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_PARAMETER_H
#define SURROGATE_TOOLKIT_PARAMETER_H

#include "parameter_range.h"

namespace parameter {

enum class ParameterCategory {
    Continuous, Discrete, Categorical, Text
};

enum class ParameterDirection {
    In, Out, Both
};

struct Parameter {
    std::string name;
    ParameterCategory category;
    ParameterDirection direction;
    virtual void ingest();
    virtual void emit();
};

template <typename T>
struct ParameterT : public Parameter {
    std::function<const T&(void)> get_readable;
    std::function<T&(void)> get_writeable;
    ranges::ParameterRange<T> range;
};


} // namespace parameter

#endif //SURROGATE_TOOLKIT_PARAMETER_H
