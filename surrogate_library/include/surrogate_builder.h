
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
#define SURROGATE_TOOLKIT_SURROGATE_BUILDER_H

#include <set>
#include <variant>
#include <vector>

#include "parameter.h"
#include "parameter_range.h"
using namespace parameter;
using namespace ranges;

void hello_from_surrogate_library();


class SurrogateModel {

    std::vector<Parameter> m_params;

public:
    void capture_training_data() {

    }
    void capture_input_ranges() {

    }
    void train_using_samples() {

    }


    template <typename T>
    void addInput(std::string name, std::function<const T&(void)> getter, ParameterCategory cat, ParameterRange<T> range) {
        ParameterT<T> param;
        param.name = std::move(name);
        param.get_readable = getter;
        param.direction = ParameterDirection::In;
        param.category = cat;
        param.range = range;
        m_params.push_back(param);
    }

    template <typename T>
    void addOutput(std::string name, std::function<T&(void)> setter, ParameterCategory cat, ParameterRange<T> range) {
        ParameterT<T> param;
        param.name = std::move(name);
        param.get_writeable = setter;
        param.direction = ParameterDirection::Out;
        param.category = cat;
        param.range = range;
        m_params.push_back(param);
    }

    template <typename T>
    void addInputOutput(std::string name, std::function<const T&(void)> setter, ParameterCategory cat, ParameterRange<T> range) {
        ParameterT<T> param;
        param.name = std::move(name);
        param.get_writeable = setter;
        param.direction = ParameterDirection::Both;
        param.category = cat;
        param.range = range;
        m_params.push_back(param);
    }
};

#endif //SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
