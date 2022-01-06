
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
    virtual size_t capture_value() = 0;
    virtual void capture_range() = 0;
    virtual void deploy_sample_value() = 0;
    virtual ~Input() = default;
};

template <typename T>
struct InputT : public Input {
    std::vector<T> captures;
    T sample;
    std::function<T*(void)> accessor;
    Range<T> range;
    RangeFinder<T> rangeFinder;


    size_t capture_value() override {
	T dest = *accessor();
	captures.push_back(dest);
	return captures.size() - 1;
    }

    void capture_range() override {
        rangeFinder.capture(*accessor());
    }

    void deploy_sample_value() override {
	*accessor() = sample;
    }
};

struct Output {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    virtual size_t capture() = 0;
    virtual ~Output() = default;
};

template <typename T>
struct OutputT : public Output {
    std::vector<T> captures;
    std::function<T(void)> getter;

    size_t capture() override {
	T val = getter();
	captures.push_back(val);
	return captures.size() - 1;
    }
};


#endif //SURROGATE_TOOLKIT_PARAMETER_H
